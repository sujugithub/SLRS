"""
Training screen — premium Apple HIG-inspired redesign.

All camera/capture/save logic is preserved exactly.
Only the visual layer is redesigned.
"""
import re
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, SAMPLES_PER_CLASS,
    SEQUENCE_DATA_DIR, SEQ_LENGTH, MIN_SEQUENCES_PER_SIGN,
    LSTM_MODEL_FILE, LSTM_LABELS_FILE,
)
from core.camera_handler import CameraHandler
from core.hand_detector import HandDetector
from core.holistic_detector import (
    HAND_CONNECTIONS as _HAND_CONN,
    FACE_CONNECTIONS as _FACE_CONN,
    POSE_CONNECTIONS as _POSE_CONN,
    POSE_VIS_THRESHOLD as _POSE_VIS,
    KEY_FACE_INDICES as _KEY_FACE,
)
from gui.design import (
    BG_DEEP, BG_BASE, BG_ELEVATED, BG_SURFACE, BG_BORDER, BG_FLOAT, BG_ACTIVE,
    ACCENT, SUCCESS, DANGER,
    TEXT_TITLE, TEXT_PRIMARY, TEXT_SEC, TEXT_HINT,
    F_TITLE, F_BODY_LB, F_BODY_B, F_BODY, F_CAP_B, F_CAP, F_SMALL, _f,
    lerp, rrect,
    GlassCard, PillButton,
)


MIN_IMAGES              = 10
MAX_IMAGES              = SAMPLES_PER_CLASS
THUMB_SIZE              = 64
CAMERA_UPDATE_MS        = 33    # ~30 fps
AUTO_CAPTURE_INTERVAL   = 300   # ms between automatic captures
AUTO_CAPTURE_TARGET     = 30    # samples to collect before auto-stop

# Dynamic mode constants
MIN_SEQS   = MIN_SEQUENCES_PER_SIGN   # minimum sequences before Save unlocks
MAX_SEQS   = 30                       # cap sequences per session


class TrainingScreen:
    """Screen for capturing training data with live camera and hand detection."""

    def __init__(self, parent, on_back=None, on_save=None,
                 on_save_sequences=None,
                 camera=None, detector=None, pose_detector=None,
                 holistic=None):
        self.parent              = parent
        self.on_back             = on_back
        self.on_save             = on_save
        self.on_save_sequences   = on_save_sequences   # (sign_name, list_of_(30,183)_arrays)

        self.captured_features   = []
        self._thumb_refs         = []
        self._photo_ref          = None
        self._update_job         = None
        self._current_landmarks  = None

        # Auto-capture state
        self._auto_capturing     = False
        self._auto_capture_job   = None

        self._camera         = camera or CameraHandler()
        self._detector       = detector or HandDetector()
        self._pose_detector  = pose_detector
        self._owns_resources = camera is None

        # ── Dynamic mode state ─────────────────────────────────────────────
        self._mode            = "static"    # "static" | "dynamic"
        self._holistic        = holistic     # HolisticDetector — shared from App
        self._owns_holistic   = False        # True only if we created it ourselves
        self.captured_seqs    = []          # list of (SEQ_LENGTH, 183) arrays
        self._recording       = False       # True while recording a sequence
        self._record_buffer   = []          # holistic features for current seq
        self._current_holistic_feats = None # latest frame's holistic features

        self.frame = tk.Frame(parent, bg=BG_DEEP)
        self._build_ui()

    # ─── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top navigation bar ────────────────────────────────────────────
        nav = tk.Frame(self.frame, bg=BG_BASE, height=56)
        nav.pack(fill=tk.X)
        nav.pack_propagate(False)
        tk.Frame(self.frame, bg=BG_BORDER, height=1).pack(fill=tk.X)

        PillButton(
            nav, text="← Home", command=self._on_back,
            color=BG_ELEVATED, hover_color=BG_FLOAT,
            height=34, btn_width=110, btn_font=F_BODY_B,
        ).pack(side=tk.LEFT, padx=(16, 0), pady=11)

        tk.Label(nav, text="Train New Sign",
                 font=F_TITLE, fg=TEXT_TITLE, bg=BG_BASE).pack(
            side=tk.LEFT, padx=18)

        self.hand_status = tk.Label(
            nav, text="", font=F_CAP, fg=TEXT_SEC, bg=BG_BASE)
        self.hand_status.pack(side=tk.RIGHT, padx=(0, 16))

        # Step indicator (right of nav)
        step_row = tk.Frame(nav, bg=BG_BASE)
        step_row.pack(side=tk.RIGHT, padx=(0, 12))
        self._step_dots = []
        for i, _ in enumerate(("Name", "Capture", "Save")):
            dot = tk.Canvas(step_row, width=8, height=8,
                            bg=BG_BASE, highlightthickness=0)
            dot.pack(side=tk.LEFT)
            dot.create_oval(0, 0, 8, 8,
                            fill=ACCENT if i == 0 else BG_FLOAT, outline="")
            self._step_dots.append(dot)
            if i < 2:
                tk.Label(step_row, text="  ―  ", font=F_SMALL,
                         fg=BG_FLOAT, bg=BG_BASE).pack(side=tk.LEFT)

        # ── Main content ──────────────────────────────────────────────────
        content = tk.Frame(self.frame, bg=BG_DEEP)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        # ── Left: camera + thumbnail strip ────────────────────────────────
        left = tk.Frame(content, bg=BG_DEEP)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Camera inside a glass card (provides border + depth)
        cam_outer = GlassCard(left, bg=BG_SURFACE, border_color=BG_BORDER)
        cam_outer.pack()
        self._cam_border = cam_outer

        self.camera_canvas = tk.Canvas(
            cam_outer.body,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            bg=BG_SURFACE,
            highlightthickness=0,
        )
        self.camera_canvas.pack()
        self._draw_placeholder()

        # Thumbnail strip label
        tk.Label(left, text="Captured Frames",
                 font=F_CAP_B, fg=TEXT_HINT, bg=BG_DEEP,
                 anchor=tk.W).pack(fill=tk.X, pady=(8, 4))

        # Thumbnail glass card
        thumb_card = GlassCard(left, bg=BG_SURFACE, border_color=BG_BORDER)
        thumb_card.pack(fill=tk.X)

        self.thumb_canvas = tk.Canvas(
            thumb_card.body,
            height=THUMB_SIZE + 12,
            bg=BG_SURFACE,
            highlightthickness=0,
        )
        self.thumb_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        thumb_scroll = ttk.Scrollbar(
            thumb_card.body, orient=tk.HORIZONTAL,
            command=self.thumb_canvas.xview)
        thumb_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.thumb_canvas.config(xscrollcommand=thumb_scroll.set)

        self.thumb_inner = tk.Frame(self.thumb_canvas, bg=BG_SURFACE)
        self.thumb_canvas.create_window(
            (0, 0), window=self.thumb_inner, anchor=tk.NW)
        self.thumb_inner.bind(
            "<Configure>",
            lambda e: self.thumb_canvas.config(
                scrollregion=self.thumb_canvas.bbox("all")),
        )

        # ── Right: controls panel ─────────────────────────────────────────
        right_card = GlassCard(content, bg=BG_ELEVATED,
                               border_color=BG_BORDER, width=264)
        right_card.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        right_card.pack_propagate(False)

        right = tk.Frame(right_card.body, bg=BG_ELEVATED)
        right.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        # ─ Mode toggle (Static | Dynamic)
        mode_row = tk.Frame(right, bg=BG_ELEVATED)
        mode_row.pack(fill=tk.X, pady=(0, 12))

        tk.Label(mode_row, text="SIGN TYPE",
                 font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED).pack(
            side=tk.LEFT, padx=(0, 8))

        self._mode_var = tk.StringVar(value="static")
        for val, label in (("static", "Static"), ("dynamic", "Dynamic")):
            tk.Radiobutton(
                mode_row, text=label, value=val,
                variable=self._mode_var,
                command=lambda v=val: self._switch_mode(v),
                bg=BG_ELEVATED, fg=TEXT_PRIMARY,
                selectcolor=BG_FLOAT,
                activebackground=BG_ELEVATED, activeforeground=ACCENT,
                font=F_CAP_B, cursor="hand2",
            ).pack(side=tk.LEFT, padx=2)

        tk.Frame(right, height=1, bg=BG_BORDER).pack(fill=tk.X, pady=(0, 10))

        # ─ Sign name input
        tk.Label(right, text="SIGN NAME",
                 font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED,
                 anchor=tk.W).pack(fill=tk.X)
        tk.Frame(right, height=5, bg=BG_ELEVATED).pack()

        entry_frame = tk.Frame(right, bg=BG_ACTIVE, pady=1)
        entry_frame.pack(fill=tk.X)
        self.name_entry = tk.Entry(
            entry_frame,
            font=F_BODY,
            bg=BG_FLOAT,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief=tk.FLAT,
            highlightthickness=0,
        )
        self.name_entry.pack(fill=tk.X, ipady=7, padx=1)
        self.name_entry.bind("<FocusIn>",
                             lambda *_: entry_frame.config(bg=ACCENT))
        self.name_entry.bind("<FocusOut>",
                             lambda *_: entry_frame.config(bg=BG_ACTIVE))

        tk.Frame(right, height=20, bg=BG_ELEVATED).pack()

        # ─ Capture counter
        count_row = tk.Frame(right, bg=BG_ELEVATED)
        count_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(count_row, text="CAPTURED",
                 font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED).pack(side=tk.LEFT)

        self.count_label = tk.Label(
            count_row, text=f"0 / {MAX_IMAGES}",
            font=F_BODY_LB, fg=TEXT_PRIMARY, bg=BG_ELEVATED)
        self.count_label.pack(side=tk.RIGHT)

        # ─ Progress bar (custom Canvas pill)
        self._prog_canvas = tk.Canvas(
            right, height=8, bg=BG_ELEVATED, highlightthickness=0)
        self._prog_canvas.pack(fill=tk.X, pady=(0, 18))
        self._prog_canvas.bind("<Configure>",
                               lambda e: self._draw_progress(len(self.captured_features)))

        # ─ Auto-Capture button (starts/stops auto-capture loop)
        self.start_capture_btn = PillButton(
            right, text="▶  Start Auto-Capture", command=self._toggle_auto_capture,
            color=ACCENT, height=44, btn_width=1, btn_font=F_BODY_B,
        )
        self.start_capture_btn.pack(fill=tk.X, pady=(0, 6))

        # ─ Live capture status ("Capturing... (12/30)" or completion msg)
        self._auto_status_label = tk.Label(
            right,
            text=f"Press Start — captures {AUTO_CAPTURE_TARGET} frames automatically",
            font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED,
            wraplength=220, justify=tk.LEFT, anchor=tk.W,
        )
        self._auto_status_label.pack(fill=tk.X, pady=(0, 10))

        # ─ Separator
        tk.Frame(right, height=1, bg=BG_BORDER).pack(fill=tk.X, pady=(0, 10))

        # ─ Save & Train button (unlocks after MIN_IMAGES)
        self.save_btn = PillButton(
            right, text="Save & Train ✓", command=self._on_save,
            color=SUCCESS, height=40, btn_width=1, btn_font=F_BODY_B,
        )
        self.save_btn.pack(fill=tk.X, pady=(0, 8))
        self.save_btn.disable()

        # ─ Cancel button
        PillButton(
            right, text="Cancel", command=self._on_cancel,
            color=BG_FLOAT, hover_color=DANGER,
            height=36, btn_width=1, btn_font=F_BODY_B,
        ).pack(fill=tk.X)

        # ─ Static controls placeholder (kept for mode-switch show/hide logic)
        self._static_extra = tk.Frame(right, bg=BG_ELEVATED)
        self._static_extra.pack(fill=tk.X)

        # ─ Dynamic controls frame (hidden by default) ─────────────────────
        self._dynamic_frame = tk.Frame(right, bg=BG_ELEVATED)
        # (not packed yet — shown when mode switches to "dynamic")

        # Sequence count row
        seq_count_row = tk.Frame(self._dynamic_frame, bg=BG_ELEVATED)
        seq_count_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(seq_count_row, text="SEQUENCES",
                 font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED).pack(side=tk.LEFT)
        self._seq_count_label = tk.Label(
            seq_count_row, text=f"0 / {MIN_SEQS} min",
            font=F_BODY_LB, fg=TEXT_PRIMARY, bg=BG_ELEVATED)
        self._seq_count_label.pack(side=tk.RIGHT)

        # Recording progress bar
        self._rec_prog_canvas = tk.Canvas(
            self._dynamic_frame, height=8, bg=BG_ELEVATED, highlightthickness=0)
        self._rec_prog_canvas.pack(fill=tk.X, pady=(0, 10))
        self._rec_prog_canvas.bind(
            "<Configure>", lambda e: self._draw_rec_progress(0.0))

        # Recording status label
        self._rec_status_label = tk.Label(
            self._dynamic_frame,
            text=f"Hold gesture and click Record ({SEQ_LENGTH} frames each)",
            font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED,
            wraplength=220, justify=tk.LEFT, anchor=tk.W,
        )
        self._rec_status_label.pack(fill=tk.X, pady=(0, 8))

        # Record button
        self._record_btn = PillButton(
            self._dynamic_frame, text="⏺  Record Sequence",
            command=self._on_record,
            color=ACCENT, height=44, btn_width=1, btn_font=F_BODY_B,
        )
        self._record_btn.pack(fill=tk.X, pady=(0, 6))

        # Dynamic separator
        tk.Frame(self._dynamic_frame, height=1, bg=BG_BORDER).pack(
            fill=tk.X, pady=(0, 8))

        # Dynamic Save & Train button
        self._dyn_save_btn = PillButton(
            self._dynamic_frame, text="Save & Train LSTM ✓",
            command=self._on_save_dynamic,
            color=SUCCESS, height=40, btn_width=1, btn_font=F_BODY_B,
        )
        self._dyn_save_btn.pack(fill=tk.X, pady=(0, 6))
        self._dyn_save_btn.disable()

        self._dyn_hint_label = tk.Label(
            self._dynamic_frame,
            text=f"Record at least {MIN_SEQS} sequences to unlock Save",
            font=F_SMALL, fg=TEXT_HINT, bg=BG_ELEVATED,
            wraplength=220, justify=tk.LEFT, anchor=tk.W,
        )
        self._dyn_hint_label.pack(fill=tk.X)

    # ─── Progress Bar ─────────────────────────────────────────────────────────
    def _draw_progress(self, count):
        """Draw a pill-shaped progress bar for capture count."""
        c = self._prog_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width() if c.winfo_width() > 1 else 220
        h = 8
        r = h // 2

        rrect(c, 0, 0, w, h, r=r, fill=BG_FLOAT)

        ratio = min(1.0, count / MAX_IMAGES)
        fill_w = int(w * ratio)
        if fill_w > h:
            color = SUCCESS if count >= MIN_IMAGES else ACCENT
            rrect(c, 0, 0, fill_w, h, r=r, fill=color)
        elif fill_w > 0:
            c.create_oval(0, 0, h, h, fill=ACCENT, outline="")

    # ─── Camera Loop ──────────────────────────────────────────────────────────
    def _start_camera(self):
        ok = self._camera.start()
        if not ok:
            self.hand_status.config(
                text="Camera not found — check connection", fg=DANGER)
            self.camera_canvas.delete("all")
            cx, cy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2
            self.camera_canvas.create_text(
                cx, cy - 12, text="No camera detected",
                font=_f(15, "bold"), fill=DANGER)
            self.camera_canvas.create_text(
                cx, cy + 14,
                text="Check your webcam connection.",
                font=F_CAP, fill=TEXT_SEC)
            return
        self.hand_status.config(text="Camera starting…", fg=TEXT_SEC)
        self._schedule_update()

    def _stop_camera(self):
        if self._update_job is not None:
            self.parent.after_cancel(self._update_job)
            self._update_job = None
        self._camera.stop()
        self._current_landmarks = None

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

            if self._mode == "dynamic":
                self._update_frame_dynamic(frame)
            else:
                self._update_frame_static(frame)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self._photo_ref = photo
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        except tk.TclError:
            return

        self._schedule_update()

    def _update_frame_static(self, frame):
        """Static mode: hand detection + thumb overlay.

        Uses the shared HolisticDetector when available so landmark drawing
        matches the prediction screen exactly (green/purple hands, orange
        body-only pose, teal face mesh).  Falls back to the legacy separate
        detectors when no holistic instance was injected.
        """
        if self._holistic is not None:
            hand_lms_list, pose_lms, face_lms = self._holistic.detect_all(frame)
            self._draw_landmarks_custom(
                frame, hand_lms_list, pose_lms, face_lms,
                handedness=self._holistic._last_handedness,
            )
            self._current_landmarks = hand_lms_list[0] if hand_lms_list else None
            if hand_lms_list:
                count = len(hand_lms_list)
                label = "\u270b Hand detected" if count == 1 else f"\u270b {count} hands"
                self.hand_status.config(text=label, fg=SUCCESS)
                self._cam_border.configure(bg=lerp(BG_BORDER, SUCCESS, 0.35))
            else:
                self.hand_status.config(text="No hand detected", fg=TEXT_SEC)
                self._cam_border.configure(bg=BG_BORDER)
        else:
            # Legacy fallback (no holistic detector)
            if self._pose_detector:
                pose_lms = self._pose_detector.detect(frame)
                if pose_lms:
                    self._pose_detector.draw_face_and_shoulders(frame, pose_lms)
            all_landmarks = self._detector.detect(frame)
            self._current_landmarks = all_landmarks[0] if all_landmarks else None
            if all_landmarks:
                for hand_lms in all_landmarks:
                    self._detector.draw_landmarks(frame, hand_lms)
                count = len(all_landmarks)
                label = "\u270b Hand detected" if count == 1 else f"\u270b {count} hands"
                self.hand_status.config(text=label, fg=SUCCESS)
                self._cam_border.configure(bg=lerp(BG_BORDER, SUCCESS, 0.35))
            else:
                self.hand_status.config(text="No hand detected", fg=TEXT_SEC)
                self._cam_border.configure(bg=BG_BORDER)

    def _update_frame_dynamic(self, frame):
        """Dynamic mode: holistic detection; collect frames when recording."""
        if self._holistic is None:
            return

        hand_lms_list, pose_lms, face_lms = self._holistic.detect_all(frame)
        self._draw_landmarks_custom(
            frame, hand_lms_list, pose_lms, face_lms,
            handedness=self._holistic._last_handedness,
        )
        self._current_holistic_feats = self._holistic.extract_holistic_features(
            hand_lms_list, pose_lms, face_lms
        )

        hand_detected = bool(hand_lms_list)
        if hand_detected:
            self.hand_status.config(text="✋ Hand detected", fg=SUCCESS)
            self._cam_border.configure(bg=lerp(BG_BORDER, SUCCESS, 0.35))
        else:
            self.hand_status.config(text="No hand detected", fg=TEXT_SEC)
            self._cam_border.configure(bg=BG_BORDER)

        # Accumulate frames for the current recording pass
        if self._recording:
            self._record_buffer.append(self._current_holistic_feats.copy())
            progress = len(self._record_buffer) / SEQ_LENGTH
            self._draw_rec_progress(progress)
            remaining = SEQ_LENGTH - len(self._record_buffer)
            self._rec_status_label.config(
                text=f"Recording… {remaining} frames remaining",
                fg=ACCENT,
            )

            if len(self._record_buffer) >= SEQ_LENGTH:
                # Sequence complete
                seq = np.array(self._record_buffer[:SEQ_LENGTH], dtype=np.float32)
                self.captured_seqs.append(seq)
                self._record_buffer.clear()
                self._recording = False
                self._record_btn.enable()
                self._update_seq_count()
                self._cam_border.configure(bg=lerp(BG_BORDER, SUCCESS, 0.6))
                self.parent.after(
                    300,
                    lambda: self._safe_config(self._cam_border, bg=BG_BORDER),
                )

    # ─── Drawing Helpers ──────────────────────────────────────────────────────
    def _draw_landmarks_custom(
        self, frame, hand_lms_list, pose_lms, face_lms, handedness=None
    ) -> None:
        """Identical drawing rules to PredictionScreen._draw_landmarks_custom.

        Hands  → right=GREEN, left=PURPLE
        Pose   → ORANGE, body/arms only (skip face landmarks 0-10)
        Face   → teal contours
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
        # 2. Pose — ORANGE, body & arms only (skip pose landmarks 0-10)
        if pose_lms:
            _ORANGE = (0, 165, 255)
            pts: dict = {}
            for idx, lm in enumerate(pose_lms):
                if idx <= 10:
                    continue
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
            dot_col  = (0, 220, 0)  if is_right else (200, 0, 220)
            line_col = (0, 160, 0)  if is_right else (140, 0, 160)
            hpts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in _HAND_CONN:
                cv2.line(frame, hpts[a], hpts[b], line_col, 2)
            for pt in hpts:
                cv2.circle(frame, pt, 4, dot_col, -1)

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

    def _add_thumbnail(self, bgr_frame):
        idx = len(self.captured_features) - 1
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(rgb)
        pil_thumb = pil_img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        photo    = ImageTk.PhotoImage(image=pil_thumb)
        self._thumb_refs.append(photo)

        tk.Label(
            self.thumb_inner,
            image=photo,
            bg=BG_SURFACE,
            highlightthickness=1,
            highlightbackground=ACCENT,
        ).grid(row=0, column=idx, padx=2, pady=4)

    # ─── Widget Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _safe_config(widget, **kwargs):
        try:
            widget.config(**kwargs)
        except tk.TclError:
            pass

    def _update_count(self):
        count = len(self.captured_features)
        self.count_label.config(text=f"{count} / {MAX_IMAGES}")
        self._draw_progress(count)
        if count >= MIN_IMAGES:
            self.save_btn.enable()

    # ─── Public API ───────────────────────────────────────────────────────────
    def show(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        self._start_camera()

    def hide(self):
        self._stop_camera()
        self.frame.pack_forget()

    def reset(self):
        self._stop_auto_capture()
        self.captured_features.clear()
        self.captured_seqs.clear()
        self._record_buffer.clear()
        self._recording = False
        self._current_landmarks = None
        self._current_holistic_feats = None
        self.name_entry.delete(0, tk.END)
        for widget in self.thumb_inner.winfo_children():
            widget.destroy()
        self._draw_placeholder()
        self._update_count()
        self._update_seq_count()
        self.save_btn.disable()
        if hasattr(self, "_auto_status_label"):
            self._auto_status_label.config(
                text=f"Press Start — captures {AUTO_CAPTURE_TARGET} frames automatically",
                fg=TEXT_HINT,
            )
        if hasattr(self, "_dyn_save_btn"):
            self._dyn_save_btn.disable()

    # ─── Callbacks ────────────────────────────────────────────────────────────
    # ─── Auto-Capture ────────────────────────────────────────────────────────
    def _toggle_auto_capture(self):
        """Toggle the auto-capture loop on/off."""
        if self._auto_capturing:
            self._stop_auto_capture()
        else:
            self._start_auto_capture()

    def _start_auto_capture(self):
        """Begin auto-capturing one frame every AUTO_CAPTURE_INTERVAL ms."""
        if self._current_landmarks is None:
            messagebox.showinfo(
                "No Hand Detected",
                "Hold your hand clearly in front of the camera\n"
                "and wait for the green 'Hand detected' indicator.",
            )
            return
        self._auto_capturing = True
        self.start_capture_btn._text = "⏹  Stop Capture"
        self.start_capture_btn._paint(self.start_capture_btn._cur)
        self._auto_capture_step()

    def _stop_auto_capture(self):
        """Cancel the auto-capture loop and reset the button label."""
        self._auto_capturing = False
        if self._auto_capture_job is not None:
            self.parent.after_cancel(self._auto_capture_job)
            self._auto_capture_job = None
        self.start_capture_btn._text = "▶  Start Auto-Capture"
        self.start_capture_btn._paint(self.start_capture_btn._cur)

    def _auto_capture_step(self):
        """Capture one frame then schedule the next step if still running."""
        if not self._auto_capturing:
            return

        count = len(self.captured_features)
        if count >= AUTO_CAPTURE_TARGET:
            # Target reached — stop and unlock Save
            self._stop_auto_capture()
            self._auto_status_label.config(
                text=f"✓ Done! {count} frames captured — ready to Save & Train",
                fg=SUCCESS,
            )
            return

        # Capture current frame if hand is visible
        if self._current_landmarks is not None:
            features = self._detector.extract_features(self._current_landmarks)
            self.captured_features.append(features)
            frame = self._camera.get_frame()
            if frame is not None:
                self._add_thumbnail(frame)
            count = len(self.captured_features)
            self._update_count()
            self._auto_status_label.config(
                text=f"Capturing...  ({count} / {AUTO_CAPTURE_TARGET})",
                fg=ACCENT,
            )
            # Brief flash on camera border
            self._cam_border.configure(bg=ACCENT)
            self.parent.after(
                100,
                lambda: self._safe_config(self._cam_border, bg=BG_BORDER),
            )
        else:
            self._auto_status_label.config(
                text=f"Waiting for hand...  ({count} / {AUTO_CAPTURE_TARGET})",
                fg=TEXT_SEC,
            )

        # Schedule next capture
        self._auto_capture_job = self.parent.after(
            AUTO_CAPTURE_INTERVAL, self._auto_capture_step
        )

    def _on_back(self):
        self._stop_auto_capture()
        self._stop_camera()
        self.reset()
        if self.on_back:
            self.on_back()

    @staticmethod
    def _sanitize_name(raw):
        name = raw.strip().replace(" ", "_").lower()
        name = re.sub(r"[^a-z0-9_]", "", name)
        return name[:50]

    def _on_save(self):
        sign_name = self._sanitize_name(self.name_entry.get())
        if not sign_name:
            messagebox.showwarning(
                "Invalid Name",
                "Please enter a sign name using letters, numbers, or spaces.\n\n"
                "Examples: 'Hello', 'Thank You', 'Stop'",
            )
            return
        if len(self.captured_features) < MIN_IMAGES:
            remaining = MIN_IMAGES - len(self.captured_features)
            messagebox.showwarning(
                "Not Enough Data",
                f"You need at least {MIN_IMAGES} samples to train.\n\n"
                f"Captured: {len(self.captured_features)}\n"
                f"Remaining: {remaining} more needed",
            )
            return
        self._stop_camera()
        if self.on_save:
            self.on_save(sign_name, self.captured_features)

    def _on_cancel(self):
        self._stop_auto_capture()
        self._stop_camera()
        self.reset()
        if self.on_back:
            self.on_back()

    def has_unsaved_data(self):
        return len(self.captured_features) > 0 or len(self.captured_seqs) > 0

    def cleanup(self):
        self._stop_camera()
        if self._owns_resources:
            self._detector.close()
        if self._holistic is not None and self._owns_holistic:
            self._holistic.close()
            self._holistic = None

    # ─── Dynamic-mode helpers ─────────────────────────────────────────────────

    def _switch_mode(self, mode: str) -> None:
        """Toggle between static and dynamic recording modes."""
        self._mode = mode
        if mode == "dynamic":
            self._stop_auto_capture()
            self._init_holistic()
            # Hide static-only widgets
            self.start_capture_btn.pack_forget()
            self._auto_status_label.pack_forget()
            self._static_extra.pack_forget()
            # Show dynamic panel
            self._dynamic_frame.pack(fill=tk.X)
            self._update_seq_count()
        else:
            # Show static widgets
            self._dynamic_frame.pack_forget()
            self.start_capture_btn.pack(fill=tk.X, pady=(0, 6))
            self._auto_status_label.pack(fill=tk.X, pady=(0, 10))
            self._static_extra.pack(fill=tk.X)

    def _init_holistic(self) -> None:
        """Lazily create the HolisticDetector on first switch to Dynamic.

        If a holistic instance was already injected from App (shared), it is
        used directly.  Only allocates a new detector when none was provided.
        """
        if self._holistic is not None:
            return
        try:
            from core.holistic_detector import HolisticDetector
            self._holistic = HolisticDetector()
            self._owns_holistic = True
        except Exception as exc:
            messagebox.showerror(
                "Holistic Model Error",
                f"Could not load holistic detector:\n{exc}\n\n"
                "Falling back to Static mode.",
            )
            self._mode_var.set("static")
            self._mode = "static"

    def _on_record(self) -> None:
        """Start recording a single gesture sequence."""
        if self._holistic is None:
            messagebox.showinfo("Not Ready", "Holistic detector is not initialised.")
            return
        if len(self.captured_seqs) >= MAX_SEQS:
            messagebox.showinfo(
                "Maximum Reached",
                f"You've recorded the maximum {MAX_SEQS} sequences.\n"
                "Click 'Save & Train LSTM' to continue.",
            )
            return
        self._recording = True
        self._record_buffer.clear()
        self._record_btn.disable()
        self._rec_status_label.config(
            text=f"Recording… {SEQ_LENGTH} frames remaining", fg=ACCENT)
        self._draw_rec_progress(0.0)

    def _draw_rec_progress(self, ratio: float) -> None:
        """Draw the recording progress bar."""
        c = self._rec_prog_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width() if c.winfo_width() > 1 else 220
        h = 8
        r = h // 2
        rrect(c, 0, 0, w, h, r=r, fill=BG_FLOAT)
        fill_w = int(w * min(1.0, ratio))
        if fill_w > h:
            color = SUCCESS if ratio >= 1.0 else ACCENT
            rrect(c, 0, 0, fill_w, h, r=r, fill=color)

    def _update_seq_count(self) -> None:
        """Refresh sequence counter and unlock/lock the dynamic Save button."""
        if not hasattr(self, "_seq_count_label"):
            return
        n = len(self.captured_seqs)
        self._seq_count_label.config(text=f"{n} / {MIN_SEQS} min")
        if n >= MIN_SEQS:
            self._dyn_save_btn.enable()
            self._dyn_hint_label.config(
                text=f"{n} sequences — ready to train LSTM!", fg=SUCCESS)
        else:
            remaining = MIN_SEQS - n
            self._dyn_hint_label.config(
                text=f"Record {remaining} more to unlock Save", fg=TEXT_HINT)

    def _on_save_dynamic(self) -> None:
        """Validate, stop camera, then invoke the on_save_sequences callback."""
        sign_name = self._sanitize_name(self.name_entry.get())
        if not sign_name:
            messagebox.showwarning(
                "Invalid Name",
                "Please enter a sign name (letters, numbers, spaces).",
            )
            return
        if len(self.captured_seqs) < MIN_SEQS:
            messagebox.showwarning(
                "Not Enough Sequences",
                f"Record at least {MIN_SEQS} sequences.\n"
                f"Current: {len(self.captured_seqs)}",
            )
            return
        self._stop_camera()
        if self.on_save_sequences:
            self.on_save_sequences(sign_name, list(self.captured_seqs))
