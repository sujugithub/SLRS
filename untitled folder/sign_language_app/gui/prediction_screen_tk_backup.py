"""
Prediction screen — premium Apple HIG-inspired redesign.

All camera/detection/sentence-builder logic is preserved exactly.
Only the visual layer (layout, colors, typography, widgets) is redesigned.
"""
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from config import CAMERA_WIDTH, CAMERA_HEIGHT
from core.camera_handler import CameraHandler
from core.hand_detector import HandDetector
from core.model_trainer import SignModel
from core.holistic_detector import (
    HAND_CONNECTIONS as _HAND_CONN,
    FACE_CONNECTIONS as _FACE_CONN,
    POSE_CONNECTIONS as _POSE_CONN,
    POSE_VIS_THRESHOLD as _POSE_VIS,
    KEY_FACE_INDICES as _KEY_FACE,
)
from core.sentence_buffer import SentenceBuffer
from core.nlp_processor import RuleBasedNLP
from core.sequence_collector import SequenceCollector
from gui.design import (
    BG_DEEP, BG_BASE, BG_SURFACE, BG_ELEVATED, BG_FLOAT, BG_BORDER, BG_ACTIVE,
    ACCENT, ACCENT_HOVER, ACCENT_DARK,
    SUCCESS, SUCCESS_TINT, WARNING, DANGER, DANGER_TINT, INFO,
    TEXT_TITLE, TEXT_PRIMARY, TEXT_SEC, TEXT_HINT,
    F_TITLE, F_HEAD, F_BODY_LB, F_BODY_L, F_BODY_B, F_BODY, F_CAP_B, F_CAP, F_SMALL, F_SIGN, _f,
    lerp, darken, _rgb, _hex, rrect, rrect_border,
    GlassCard, PillButton, StatusDot,
)


CAMERA_UPDATE_MS    = 33    # ~30 fps
CONFIDENCE_THRESHOLD = 0.50
AUTO_ADD_FRAMES     = 30    # ~1.0 s at 30 fps
COOLDOWN_FRAMES     = 60


class PredictionScreen:
    """Screen for real-time sign language prediction with live camera."""

    def __init__(self, parent, model, on_back=None, camera=None,
                 detector=None, pose_detector=None, speaker=None,
                 holistic=None, lstm_model=None):
        self.parent        = parent
        self.model         = model
        self.on_back       = on_back
        self._speaker      = speaker
        self._pose_detector = pose_detector

        self._photo_ref  = None
        self._update_job = None

        self._camera   = camera or CameraHandler()
        self._detector = detector or HandDetector()
        self._owns_resources = camera is None

        self._sentence_buffer = SentenceBuffer()
        self._nlp             = RuleBasedNLP()

        self._stable_count      = 0
        self._stable_sign       = None
        self._cooldown          = 0
        self._current_prediction = None
        self._tts_muted          = False

        # ── LSTM / dynamic prediction ─────────────────────────────────────
        self._holistic    = holistic      # HolisticDetector (shared from App)
        self._lstm_model  = lstm_model    # LSTMSignModel (shared from App)
        self._seq_collector = SequenceCollector(seq_len=30)

        self.frame = tk.Frame(parent, bg=BG_DEEP)
        self._build_ui()

    # ─── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top navigation bar ────────────────────────────────────────────
        nav = tk.Frame(self.frame, bg=BG_BASE, height=56)
        nav.pack(fill=tk.X)
        nav.pack_propagate(False)

        # 1px bottom border on nav
        tk.Frame(self.frame, bg=BG_BORDER, height=1).pack(fill=tk.X)

        # Back button
        self._back_btn = PillButton(
            nav, text="← Home", command=self._on_back,
            color=BG_ELEVATED, hover_color=BG_FLOAT,
            height=34, btn_width=110, btn_font=F_BODY_B,
        )
        self._back_btn.pack(side=tk.LEFT, padx=(16, 0), pady=11)

        # Screen title
        tk.Label(nav, text="Live Detection",
                 font=F_TITLE, fg=TEXT_TITLE, bg=BG_BASE).pack(
            side=tk.LEFT, padx=18)

        # Hand status indicator (right side of nav)
        self.hand_status = tk.Label(
            nav, text="", font=F_CAP, fg=TEXT_SEC, bg=BG_BASE)
        self.hand_status.pack(side=tk.RIGHT, padx=(0, 16))

        # Mute toggle
        self.mute_btn = PillButton(
            nav, text="🔊  Mute", command=self._toggle_mute,
            color=BG_ELEVATED, hover_color=BG_FLOAT,
            height=34, btn_width=110, btn_font=F_BODY_B,
        )
        self.mute_btn.pack(side=tk.RIGHT, padx=(0, 10), pady=11)

        # ── Main content area ─────────────────────────────────────────────
        content = tk.Frame(self.frame, bg=BG_DEEP)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        # ── Left: camera feed ─────────────────────────────────────────────
        left = tk.Frame(content, bg=BG_DEEP)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Camera glass frame (provides glowing border effect)
        cam_outer = GlassCard(left, bg=BG_SURFACE, border_color=BG_BORDER)
        cam_outer.pack()
        self._cam_border = cam_outer  # store ref to change border color on detection

        self.camera_canvas = tk.Canvas(
            cam_outer.body,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            bg=BG_SURFACE,
            highlightthickness=0,
        )
        self.camera_canvas.pack()
        self._draw_placeholder()

        # Model info below camera
        self.model_info = tk.Label(
            left, text="", font=F_SMALL, fg=TEXT_HINT, bg=BG_DEEP,
            anchor=tk.W, justify=tk.LEFT, wraplength=CAMERA_WIDTH,
        )
        self.model_info.pack(fill=tk.X, pady=(6, 0))

        # ── Right: detection panel ────────────────────────────────────────
        right = tk.Frame(content, bg=BG_DEEP, width=268)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        right.pack_propagate(False)

        # ─ Sign display card
        sign_card = GlassCard(right, bg=BG_ELEVATED, border_color=BG_BORDER)
        sign_card.pack(fill=tk.X, pady=(0, 10))

        sign_body = tk.Frame(sign_card.body, bg=BG_ELEVATED)
        sign_body.pack(fill=tk.X, padx=14, pady=12)

        tk.Label(sign_body, text="DETECTED SIGN",
                 font=F_SMALL, fg=TEXT_HINT,
                 bg=BG_ELEVATED, anchor=tk.W).pack(fill=tk.X)

        tk.Frame(sign_body, height=6, bg=BG_ELEVATED).pack()

        # Canvas for sign display with glow effect
        self._sign_canvas = tk.Canvas(
            sign_body, height=90, bg=BG_ELEVATED, highlightthickness=0)
        self._sign_canvas.pack(fill=tk.X)
        self._sign_canvas.bind(
            "<Configure>", lambda e: self._redraw_sign())
        self._sign_text  = "---"
        self._sign_color = TEXT_HINT
        self._draw_sign_display("---", TEXT_HINT)

        # ─ Confidence card
        conf_card = GlassCard(right, bg=BG_ELEVATED, border_color=BG_BORDER)
        conf_card.pack(fill=tk.X, pady=(0, 10))

        conf_body = tk.Frame(conf_card.body, bg=BG_ELEVATED)
        conf_body.pack(fill=tk.X, padx=14, pady=12)

        conf_header = tk.Frame(conf_body, bg=BG_ELEVATED)
        conf_header.pack(fill=tk.X)

        tk.Label(conf_header, text="CONFIDENCE",
                 font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED,
                 anchor=tk.W).pack(side=tk.LEFT)

        self.confidence_label = tk.Label(
            conf_header, text="---%",
            font=F_BODY_LB, fg=TEXT_SEC, bg=BG_ELEVATED)
        self.confidence_label.pack(side=tk.RIGHT)

        tk.Frame(conf_body, height=8, bg=BG_ELEVATED).pack()

        # Custom pill confidence bar
        self.conf_bar = tk.Canvas(
            conf_body, height=10, bg=BG_ELEVATED, highlightthickness=0)
        self.conf_bar.pack(fill=tk.X)
        self._draw_confidence_bar(0.0)

        # ─ Status card
        status_card = GlassCard(right, bg=BG_ELEVATED, border_color=BG_BORDER)
        status_card.pack(fill=tk.X, pady=(0, 10))

        status_body = tk.Frame(status_card.body, bg=BG_ELEVATED)
        status_body.pack(fill=tk.X, padx=14, pady=10)

        tk.Label(status_body, text="STATUS",
                 font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED,
                 anchor=tk.W).pack(fill=tk.X)
        tk.Frame(status_body, height=4, bg=BG_ELEVATED).pack()

        # Status row: dot + label
        s_row = tk.Frame(status_body, bg=BG_ELEVATED)
        s_row.pack(fill=tk.X)

        self._status_dot = StatusDot(s_row, color=TEXT_HINT, size=8)
        self._status_dot.pack(side=tk.LEFT)

        # We reuse self.hand_status label text but need a second display here
        self._status_detail = tk.Label(
            s_row, text="Initializing…",
            font=F_CAP, fg=TEXT_SEC, bg=BG_ELEVATED)
        self._status_detail.pack(side=tk.LEFT, padx=(6, 0))

        # ── LSTM dynamic prediction card (right panel) ───────────────────
        self._build_lstm_card(right)

        # ── Sentence builder panel ────────────────────────────────────────
        self._build_sentence_panel()

    # ─── LSTM card builder ────────────────────────────────────────────────────
    def _build_lstm_card(self, parent) -> None:
        """Small card in the right panel showing LSTM / dynamic prediction."""
        if not self._holistic:
            return  # No holistic detector → skip this card

        lstm_card = GlassCard(parent, bg=BG_ELEVATED, border_color=BG_BORDER)
        lstm_card.pack(fill=tk.X, pady=(0, 10))

        body = tk.Frame(lstm_card.body, bg=BG_ELEVATED)
        body.pack(fill=tk.X, padx=14, pady=10)

        # Header row
        hdr = tk.Frame(body, bg=BG_ELEVATED)
        hdr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(hdr, text="DYNAMIC (LSTM)",
                 font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED,
                 anchor=tk.W).pack(side=tk.LEFT)
        self._lstm_status_dot = StatusDot(hdr, color=TEXT_HINT, size=7)
        self._lstm_status_dot.pack(side=tk.RIGHT)

        # Sequence fill progress bar
        self._seq_fill_bar = tk.Canvas(
            body, height=6, bg=BG_FLOAT, highlightthickness=0)
        self._seq_fill_bar.pack(fill=tk.X, pady=(0, 6))

        # LSTM sign label
        self._lstm_sign_label = tk.Label(
            body, text="---",
            font=_f(20, "bold"), fg=TEXT_HINT, bg=BG_ELEVATED,
            anchor=tk.W)
        self._lstm_sign_label.pack(fill=tk.X)

        # Confidence sub-label
        self._lstm_conf_label = tk.Label(
            body, text="", font=F_SMALL, fg=TEXT_SEC, bg=BG_ELEVATED,
            anchor=tk.W)
        self._lstm_conf_label.pack(fill=tk.X)

    # ─── LSTM display helpers ─────────────────────────────────────────────────
    def _update_lstm_seq_bar(self, ratio: float) -> None:
        """Redraw the sequence-fill progress bar (0.0–1.0)."""
        if not hasattr(self, "_seq_fill_bar"):
            return
        try:
            c = self._seq_fill_bar
            c.delete("all")
            c.update_idletasks()
            w = c.winfo_width() if c.winfo_width() > 1 else 220
            fill = int(w * max(0.0, min(1.0, ratio)))
            c.create_rectangle(0, 0, w, 6, fill=BG_FLOAT, outline="")
            if fill > 0:
                color = ACCENT if ratio < 1.0 else SUCCESS
                c.create_rectangle(0, 0, fill, 6, fill=color, outline="")
        except tk.TclError:
            pass

    def _show_lstm_prediction(self, sign: str, confidence: float) -> None:
        """Update the LSTM card with a new prediction."""
        if not hasattr(self, "_lstm_sign_label"):
            return
        try:
            LSTM_THRESHOLD = 0.55
            if sign and confidence >= LSTM_THRESHOLD:
                display = sign.replace("_", " ").title()
                color = SUCCESS if confidence >= 0.80 else ACCENT
                self._lstm_sign_label.config(text=display, fg=color)
                self._lstm_conf_label.config(
                    text=f"{confidence:.0%}", fg=TEXT_SEC)
                if hasattr(self, "_lstm_status_dot"):
                    self._lstm_status_dot.set_color(color)
            else:
                self._lstm_sign_label.config(text="---", fg=TEXT_HINT)
                self._lstm_conf_label.config(text="", fg=TEXT_SEC)
                if hasattr(self, "_lstm_status_dot"):
                    self._lstm_status_dot.set_color(TEXT_HINT)
        except tk.TclError:
            pass

    # ─── Sentence Builder UI ──────────────────────────────────────────────────
    def _build_sentence_panel(self):
        """Sentence builder glass panel docked at the bottom."""
        panel = GlassCard(self.frame, bg=BG_ELEVATED, border_color=BG_BORDER)
        panel.pack(fill=tk.X, padx=20, pady=(0, 14))

        body = panel.body
        body.config(padx=14, pady=10)

        # ─ Header row
        hdr = tk.Frame(body, bg=BG_ELEVATED)
        hdr.pack(fill=tk.X, pady=(0, 8))

        tk.Label(hdr, text="Sentence Builder",
                 font=F_HEAD, fg=TEXT_TITLE, bg=BG_ELEVATED).pack(side=tk.LEFT)

        self._auto_add_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            hdr,
            text="Auto-add  (hold ~1.5 s)",
            variable=self._auto_add_var,
            bg=BG_ELEVATED, fg=TEXT_SEC,
            selectcolor=BG_FLOAT,
            activebackground=BG_ELEVATED, activeforeground=TEXT_SEC,
            font=F_CAP, cursor="hand2",
        ).pack(side=tk.RIGHT)

        # ─ Progress + words row
        row1 = tk.Frame(body, bg=BG_ELEVATED)
        row1.pack(fill=tk.X, pady=(0, 6))

        # Hold-progress sub-panel
        prog = tk.Frame(row1, bg=BG_ELEVATED, width=170)
        prog.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))
        prog.pack_propagate(False)

        self._progress_label = tk.Label(
            prog, text="Hold: ---",
            font=F_SMALL, fg=TEXT_SEC, bg=BG_ELEVATED, anchor=tk.W)
        self._progress_label.pack(fill=tk.X)

        self._progress_bar_canvas = tk.Canvas(
            prog, height=6, bg=BG_FLOAT, highlightthickness=0)
        self._progress_bar_canvas.pack(fill=tk.X, pady=(4, 0))

        # Words display
        words_col = tk.Frame(row1, bg=BG_ELEVATED)
        words_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(words_col, text="Words",
                 font=F_CAP_B, fg=TEXT_HINT, bg=BG_ELEVATED,
                 anchor=tk.W).pack(fill=tk.X)

        self._raw_sentence_label = tk.Label(
            words_col,
            text="(empty — hold a gesture or press Add Word)",
            font=F_BODY, fg=TEXT_SEC, bg=BG_ELEVATED,
            anchor=tk.W, wraplength=420, justify=tk.LEFT,
        )
        self._raw_sentence_label.pack(fill=tk.X)

        # ─ Natural sentence row
        row2 = tk.Frame(body, bg=BG_ELEVATED)
        row2.pack(fill=tk.X, pady=(0, 8))

        tk.Label(row2, text="Natural",
                 font=F_CAP_B, fg=TEXT_HINT, bg=BG_ELEVATED,
                 anchor=tk.W).pack(side=tk.LEFT, padx=(0, 10))

        self._corrected_label = tk.Label(
            row2, text="",
            font=_f(13, "italic"), fg=ACCENT, bg=BG_ELEVATED,
            anchor=tk.W, wraplength=460, justify=tk.LEFT,
        )
        self._corrected_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ─ Action buttons row
        row3 = tk.Frame(body, bg=BG_ELEVATED)
        row3.pack(fill=tk.X)

        btn_specs = [
            ("+ Add Word",     self._add_word_manual,  ACCENT,      None),
            ("↩ Undo",         self._undo_word,         BG_FLOAT,    None),
            ("✕ Clear",        self._clear_sentence,    DANGER,      None),
            ("🔊 Speak",       self._speak_sentence,    SUCCESS,     None),
        ]
        for text, cmd, color, hover in btn_specs:
            PillButton(
                row3, text=text, command=cmd,
                color=color, hover_color=hover,
                height=34, btn_width=1, btn_font=F_CAP_B,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

    # ─── Sign Display (Canvas with glow) ──────────────────────────────────────
    def _draw_sign_display(self, text, color):
        """Draw the sign name with a colored glow on the sign canvas."""
        c = self._sign_canvas
        c.delete("all")
        w = c.winfo_width()
        if w < 4:
            w = 240
        h = 90
        r_bg, g_bg, b_bg = _rgb(BG_ELEVATED)

        # Background
        c.create_rectangle(0, 0, w, h, fill=BG_ELEVATED, outline="")

        # Glow halo when a real sign is detected
        if text not in ("---", "?"):
            r_c, g_c, b_c = _rgb(color)
            for i in range(5, 0, -1):
                t = i / 8
                gc = _hex(r_bg + (r_c - r_bg) * t * 0.25,
                          g_bg + (g_c - g_bg) * t * 0.25,
                          b_bg + (b_c - b_bg) * t * 0.25)
                pad = (5 - i) * 4
                c.create_rectangle(pad, pad, w - pad, h - pad,
                                   fill=gc, outline="")

        # Sign text
        c.create_text(w // 2, h // 2, text=text,
                      font=F_SIGN, fill=color, anchor="center")

        self._sign_text  = text
        self._sign_color = color

    def _redraw_sign(self):
        self._draw_sign_display(self._sign_text, self._sign_color)

    # ─── Camera Loop ──────────────────────────────────────────────────────────
    def _start_camera(self):
        ok = self._camera.start()
        if not ok:
            self._update_status("Camera not found", DANGER)
            self.camera_canvas.delete("all")
            cx, cy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2
            self.camera_canvas.create_text(
                cx, cy - 12, text="No camera detected",
                font=F_HEAD, fill=DANGER)
            self.camera_canvas.create_text(
                cx, cy + 16,
                text="Check your webcam connection.",
                font=F_CAP, fill=TEXT_SEC)
            return
        self._update_status("Camera starting…", TEXT_SEC)
        self._update_model_info()
        self._schedule_update()

    def _stop_camera(self):
        if self._update_job is not None:
            self.parent.after_cancel(self._update_job)
            self._update_job = None
        self._camera.stop()

    def _schedule_update(self):
        self._update_job = self.parent.after(CAMERA_UPDATE_MS, self._update_frame)

    def _update_frame(self):
        if not self._camera.is_running():
            return
        try:
            frame = self._camera.get_frame()
            if frame is None:
                self._schedule_update()
                return

            hand_lms_list: list = []
            pose_lms = None
            face_lms = None

            if self._holistic:
                # Single holistic call serves visualization, RF prediction,
                # LSTM inference, and ME-sign detection — no duplicate runs.
                hand_lms_list, pose_lms, face_lms = self._holistic.detect_all(frame)
                self._draw_landmarks_custom(
                    frame, hand_lms_list, pose_lms, face_lms,
                    handedness=self._holistic._last_handedness,
                )
            else:
                # Fallback: legacy separate detectors
                if self._pose_detector:
                    p = self._pose_detector.detect(frame)
                    if p:
                        self._pose_detector.draw_face_and_shoulders(frame, p)
                        pose_lms = p
                raw_hands = self._detector.detect(frame)
                if raw_hands:
                    for lms in raw_hands:
                        self._detector.draw_landmarks(frame, lms)
                    hand_lms_list = raw_hands

            # ── Static RF prediction ──────────────────────────────────────
            if hand_lms_list:
                self._cam_border.configure(bg=lerp(BG_BORDER, SUCCESS, 0.4))
                features = self._detector.extract_features(hand_lms_list[0])
                sign_name, confidence = self.model.predict(features)

                if sign_name and confidence >= 0.80:
                    self._show_prediction(sign_name, confidence, tier="high")
                elif sign_name and confidence >= 0.50:
                    self._show_prediction(sign_name, confidence, tier="med")
                else:
                    self._show_low_confidence(confidence)

                # ME sign overlay — only when holistic pose is available
                me_fired = False
                if pose_lms is not None:
                    me_fired = self._check_and_draw_me(frame, hand_lms_list, pose_lms)
                if not me_fired and sign_name:
                    self._draw_hud(frame, sign_name, confidence)
            else:
                self._cam_border.configure(bg=BG_BORDER)
                self._show_no_hand()

            # ── LSTM rolling-window inference ─────────────────────────────
            if self._holistic:
                feats = self._holistic.extract_holistic_features(
                    hand_lms_list, pose_lms, face_lms)
                self._seq_collector.add(feats)
                self._update_lstm_seq_bar(self._seq_collector.fill_ratio())
                if (self._seq_collector.is_ready()
                        and self._lstm_model
                        and self._lstm_model.is_trained()):
                    seq = self._seq_collector.get_sequence()
                    lstm_sign, lstm_conf = self._lstm_model.predict(seq)
                    self._show_lstm_prediction(lstm_sign, lstm_conf)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self._photo_ref = photo
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        except tk.TclError:
            return

        self._schedule_update()

    # ─── Display Updates ──────────────────────────────────────────────────────
    def _update_status(self, text, color):
        """Update the hand status in both the nav bar and status card."""
        try:
            self.hand_status.config(text=text, fg=color)
            self._status_detail.config(text=text, fg=color)
            self._status_dot.set_color(color)
        except tk.TclError:
            pass

    def _show_prediction(self, sign_name, confidence, tier: str = "high"):
        display_name = sign_name.replace("_", " ").title()
        pct_int = int(confidence * 100)
        if tier == "high":
            color       = SUCCESS
            status_text = f"\u2705 {display_name}  ({pct_int}%)"
        else:  # med (0.50–0.79)
            color       = WARNING
            status_text = f"\u26a0\ufe0f {display_name}?  ({pct_int}%) \u2014 move closer"
        self._draw_sign_display(display_name, color)
        self.confidence_label.config(text=f"{confidence:.0%}", fg=color)
        self._draw_confidence_bar(confidence)
        self._update_status(status_text, color)
        self._current_prediction = sign_name
        if tier == "high" and self._speaker:
            self._speaker.say(display_name)
        self._check_auto_add(sign_name)

    def _show_low_confidence(self, confidence):
        self._draw_sign_display("?", WARNING)
        self.confidence_label.config(text=f"{confidence:.0%}", fg=WARNING)
        self._draw_confidence_bar(confidence)
        self._update_status("\u2753 Uncertain \u2014 reposition your hand", WARNING)
        self._stable_count = 0
        self._stable_sign  = None
        self._update_auto_add_progress(0.0, None)

    def _show_no_hand(self):
        self._draw_sign_display("---", TEXT_HINT)
        self.confidence_label.config(text="---%", fg=TEXT_SEC)
        self._draw_confidence_bar(0.0)
        self._update_status("\U0001f44b Show your hand to the camera", TEXT_SEC)
        self._stable_count       = 0
        self._stable_sign        = None
        self._current_prediction = None
        self._update_auto_add_progress(0.0, None)

    def _confidence_color(self, confidence):
        if confidence >= 0.80:
            return SUCCESS
        if confidence >= 0.50:
            return WARNING
        return DANGER

    # ─── Frame overlay helpers ────────────────────────────────────────────────
    def _draw_landmarks_custom(
        self, frame, hand_lms_list, pose_lms, face_lms, handedness=None
    ) -> None:
        """Draw all landmarks with required color scheme, in-place.

        Hands  → right=GREEN, left=PURPLE
        Pose   → ORANGE, body/arms only (skip face landmarks 0-10)
        Face   → teal contours (default MediaPipe face-mesh style)
        """
        h, w = frame.shape[:2]

        # 1. Face mesh — teal contour lines
        if face_lms:
            _TEAL = (0, 210, 190)
            for i, j in _FACE_CONN:
                if i < len(face_lms) and j < len(face_lms):
                    pi = (int(face_lms[i].x * w), int(face_lms[i].y * h))
                    pj = (int(face_lms[j].x * w), int(face_lms[j].y * h))
                    cv2.line(frame, pi, pj, _TEAL, 1)
            for idx in _KEY_FACE:
                if idx < len(face_lms):
                    cv2.circle(
                        frame,
                        (int(face_lms[idx].x * w), int(face_lms[idx].y * h)),
                        2, _TEAL, -1,
                    )

        # 2. Pose — ORANGE, body & arms only (skip indices 0-10 = face region)
        if pose_lms:
            _ORANGE = (0, 165, 255)   # BGR for orange
            pts: dict = {}
            for idx, lm in enumerate(pose_lms):
                if idx <= 10:
                    continue                     # skip face landmarks
                vis = getattr(lm, "visibility", 1.0)
                if vis >= _POSE_VIS:
                    pts[idx] = (int(lm.x * w), int(lm.y * h))
            for i, j in _POSE_CONN:
                if i > 10 and j > 10 and i in pts and j in pts:
                    cv2.line(frame, pts[i], pts[j], _ORANGE, 2)
            for pt in pts.values():
                cv2.circle(frame, pt, 4, _ORANGE, -1)

        # 3. Hands — RIGHT=GREEN, LEFT=PURPLE
        for idx, hand_lms in enumerate(hand_lms_list):
            is_right = (
                handedness
                and idx < len(handedness)
                and handedness[idx]
                and handedness[idx][0].category_name.lower() == "right"
            )
            dot_col  = (0, 220, 0)   if is_right else (200, 0, 220)
            line_col = (0, 160, 0)   if is_right else (140, 0, 160)
            hpts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in _HAND_CONN:
                cv2.line(frame, hpts[a], hpts[b], line_col, 2)
            for pt in hpts:
                cv2.circle(frame, pt, 4, dot_col, -1)

    def _check_and_draw_me(self, frame, hand_lms_list, pose_lms) -> bool:
        """Draw ME-sign overlay if the index fingertip is near the chest.

        Returns True when the ME sign fires, False otherwise.
        """
        try:
            if not self.model.is_me_sign(hand_lms_list, pose_lms):
                return False
        except Exception:
            return False
        h, w = frame.shape[:2]
        tip = hand_lms_list[0][8]
        tx, ty = int(tip.x * w), int(tip.y * h)
        cv2.circle(frame, (tx, ty), 18, (0, 255, 0), 2)
        cv2.putText(frame, "ME!", (tx + 22, ty + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        self._draw_hud(frame, "ME", 1.0)
        return True

    def _draw_hud(self, frame, sign_name: str, confidence: float) -> None:
        """Render a translucent dark HUD panel in the top-left of *frame*."""
        if not sign_name:
            return
        panel_w, panel_h = 220, 64
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        color = (0, 200, 0) if confidence >= 0.80 else (0, 200, 255)
        label = sign_name.replace("_", " ").upper()
        cv2.putText(frame, label, (16, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{int(confidence * 100)}%", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_confidence_bar(self, confidence):
        """Draw a pill-shaped progress bar for confidence."""
        bar = self.conf_bar
        bar.delete("all")
        bar.update_idletasks()
        w = bar.winfo_width()
        if w <= 1:
            w = 220
        h = 10
        r = h // 2

        fill_w = int(w * max(0.0, min(1.0, confidence)))
        color  = (self._confidence_color(confidence)
                  if confidence >= CONFIDENCE_THRESHOLD else WARNING)

        # Track (background pill)
        rrect(bar, 0, 0, w, h, r=r, fill=BG_FLOAT)

        # Filled portion
        if fill_w > h:
            rrect(bar, 0, 0, fill_w, h, r=r, fill=color)
        elif fill_w > 0:
            bar.create_oval(0, 0, h, h, fill=color, outline="")

    # ─── Auto-add Logic ───────────────────────────────────────────────────────
    def _check_auto_add(self, sign_name):
        if not self._auto_add_var.get():
            self._stable_count = 0
            self._stable_sign  = None
            self._update_auto_add_progress(0.0, None)
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            self._update_auto_add_progress(0.0, None)
            return
        if sign_name == self._stable_sign:
            self._stable_count += 1
        else:
            self._stable_sign  = sign_name
            self._stable_count = 1
        ratio = min(1.0, self._stable_count / AUTO_ADD_FRAMES)
        self._update_auto_add_progress(ratio, sign_name)
        if self._stable_count >= AUTO_ADD_FRAMES:
            if self._sentence_buffer.add_word(sign_name):
                self._update_sentence_display()
            self._stable_count = 0
            self._stable_sign  = None
            self._cooldown     = COOLDOWN_FRAMES

    def _update_auto_add_progress(self, ratio, sign_name):
        if not hasattr(self, "_progress_bar_canvas"):
            return
        try:
            if sign_name and ratio > 0:
                display = sign_name.replace("_", " ").title()
                self._progress_label.config(
                    text=f"Hold: {display}  {int(ratio * 100)}%",
                    fg=TEXT_PRIMARY)
            else:
                self._progress_label.config(text="Hold: ---", fg=TEXT_SEC)

            c = self._progress_bar_canvas
            c.delete("all")
            c.update_idletasks()
            w = c.winfo_width() if c.winfo_width() > 1 else 150
            fill = int(w * max(0.0, min(1.0, ratio)))
            c.create_rectangle(0, 0, w, 6, fill=BG_FLOAT, outline="")
            if fill > 0:
                bar_color = SUCCESS if ratio >= 1.0 else ACCENT
                c.create_rectangle(0, 0, fill, 6, fill=bar_color, outline="")
        except tk.TclError:
            pass

    # ─── Sentence Builder Callbacks ───────────────────────────────────────────
    def _add_word_manual(self):
        if self._current_prediction:
            if self._sentence_buffer.add_word(self._current_prediction):
                self._stable_count = 0
                self._cooldown     = COOLDOWN_FRAMES
                self._update_sentence_display()

    def _undo_word(self):
        if self._sentence_buffer.undo():
            self._update_sentence_display()

    def _clear_sentence(self):
        self._sentence_buffer.clear()
        self._stable_count = 0
        self._stable_sign  = None
        self._cooldown     = 0
        self._update_sentence_display()

    def _speak_sentence(self):
        if self._speaker and not self._sentence_buffer.is_empty():
            words     = self._sentence_buffer.get_words()
            corrected = self._nlp.process(words)
            if corrected:
                self._speaker.speak_sentence(corrected)

    def _update_sentence_display(self):
        if not hasattr(self, "_raw_sentence_label"):
            return
        try:
            words = self._sentence_buffer.get_words()
            if words:
                self._raw_sentence_label.config(
                    text=" ".join(words), fg=TEXT_PRIMARY)
                corrected = self._nlp.process(words)
                self._corrected_label.config(text=corrected, fg=ACCENT)
            else:
                self._raw_sentence_label.config(
                    text="(empty — hold a gesture or press Add Word)",
                    fg=TEXT_SEC)
                self._corrected_label.config(text="", fg=ACCENT)
        except tk.TclError:
            pass

    def _update_model_info(self):
        signs = self.model.get_all_signs()
        if signs:
            sign_list = ", ".join(s.replace("_", " ") for s in signs)
            self.model_info.config(
                text=f"Model ready  ·  {len(signs)} signs: {sign_list}",
                fg=TEXT_HINT)
        else:
            self.model_info.config(
                text="No model trained yet — go back and train a sign first.",
                fg=DANGER)

    # ─── Drawing Helpers ──────────────────────────────────────────────────────
    def _draw_placeholder(self):
        self.camera_canvas.delete("all")
        cx, cy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2
        self.camera_canvas.create_text(
            cx, cy - 14, text="Camera Preview",
            font=_f(16, "bold"), fill=TEXT_HINT)
        self.camera_canvas.create_text(
            cx, cy + 14,
            text="Feed activates when this screen opens",
            font=F_CAP, fill=lerp(TEXT_HINT, BG_SURFACE, 0.4))

    # ─── TTS Toggle ───────────────────────────────────────────────────────────
    def _toggle_mute(self):
        self._tts_muted = not self._tts_muted
        if self._speaker:
            self._speaker.set_enabled(not self._tts_muted)
        self.mute_btn.set_text("🔇  Unmute" if self._tts_muted else "🔊  Mute")

    # ─── Public API ───────────────────────────────────────────────────────────
    def show(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        self._update_model_info()
        self._start_camera()

    def hide(self):
        self._stop_camera()
        self._show_no_hand()
        self._stable_count       = 0
        self._stable_sign        = None
        self._cooldown           = 0
        self._current_prediction = None
        self._seq_collector.clear()
        if self._speaker:
            self._speaker.reset()
        self.frame.pack_forget()

    def _on_back(self):
        self._stop_camera()
        if self.on_back:
            self.on_back()

    def cleanup(self):
        self._stop_camera()
        if self._owns_resources:
            self._detector.close()
