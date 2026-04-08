"""
Prediction screen — PyQt6, redesigned dark editorial layout.

Camera capture + MediaPipe detection run in CameraWorker (QThread). The main
thread receives annotated frames and detection data via signal, then runs the
lightweight RF / LSTM inference and updates the UI.

Visual layout
─────────────
* Massive centred prediction display (the recognised sign).
* Animated confidence bar with red→amber→green→teal colour shift.
* Live camera feed embedded as a smaller, framed panel on the right.
* Live phrase / sentence buffer displayed below the prediction in a lighter
  weight.
* Status indicator (small dot + text) reflecting detector state.

All ML logic, signals, methods, and the public API are unchanged from the
prior implementation — only the visual layout was redesigned.
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from config import CAMERA_WIDTH, CAMERA_HEIGHT
from core.camera_handler import CameraHandler
from core.feature_extractor import SpatialFeatureExtractor
from core.hand_detector import HandDetector
from core.camera_worker import CameraWorker
from core.sentence_buffer import SentenceBuffer
from core.nlp_processor import RuleBasedNLP
from core.sequence_collector import SequenceCollector
from core.temporal_smoother import TemporalSmoother
from gui.design import (
    ACCENT, BG_BASE, BG_BORDER, BG_DEEP, BG_ELEVATED, BG_SURFACE, BG_FLOAT,
    DANGER, SUCCESS, WARNING, TEXT_HINT, TEXT_PRIMARY, TEXT_SEC, TEXT_TITLE,
    HAIR, HAIR_STRONG, FONT_MONO, FONT_DISPLAY,
    AnimatedConfidenceBar, CameraFrame, GlassCard, HSep, PillButton,
    DangerButton, OutlineButton, PulsingDot, SectionLabel, StatusDot,
    cv2_to_qpixmap, lerp, confidence_color,
)

AUTO_ADD_FRAMES = 10
COOLDOWN_FRAMES = 60
_LSTM_THRESHOLD = 0.55

_CAM_W = 460
_CAM_H = 345


class PredictionScreen(QWidget):
    """Screen for real-time sign language prediction with embedded camera."""

    def __init__(self, model=None, on_back=None, camera=None,
                 detector=None, pose_detector=None, speaker=None,
                 holistic=None, lstm_model=None, parent=None):
        super().__init__(parent)
        self.model          = model
        self.on_back        = on_back
        self._speaker       = speaker
        self._pose_detector = pose_detector

        self._camera   = camera   or CameraHandler()
        self._detector = detector or HandDetector()
        self._owns_resources = camera is None

        self._sentence_buffer = SentenceBuffer()
        self._nlp             = RuleBasedNLP()

        self._stable_count       = 0
        self._stable_sign        = None
        self._cooldown           = 0
        self._current_prediction = None
        self._tts_muted          = False
        self._frame_count        = 0

        self._holistic          = holistic
        self._lstm_model        = lstm_model
        self._seq_collector     = SequenceCollector(seq_len=30)
        self._spatial_extractor = SpatialFeatureExtractor()

        self._confidence_threshold = 0.6
        self._smoothing_window     = 5
        self._smoother             = TemporalSmoother(
            window=self._smoothing_window, min_vote_share=0.55,
        )
        self._phrase_matcher = None

        self._worker: CameraWorker | None = None

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Main content area — splits horizontally between hero (left) and
        # the camera + telemetry (right).
        content = QWidget()
        content.setStyleSheet(f"background-color: {BG_DEEP};")
        content_lay = QHBoxLayout(content)
        content_lay.setContentsMargins(40, 24, 40, 18)
        content_lay.setSpacing(28)
        root.addWidget(content, 1)

        # ── Left hero column ─────────────────────────────────────────────
        hero_col = QVBoxLayout()
        hero_col.setSpacing(0)
        hero_col.setContentsMargins(0, 0, 0, 0)

        # Status row
        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        status_row.setContentsMargins(0, 0, 0, 0)
        self._status_dot_anim = PulsingDot(
            on_color=TEXT_HINT, off_color="#1a1f30", size=8,
        )
        status_row.addWidget(self._status_dot_anim)
        self._status_lbl = QLabel("INITIALIZING")
        self._status_lbl.setStyleSheet(
            f"color: {TEXT_SEC}; font-size: 11px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 2px; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        status_row.addWidget(self._status_lbl)
        status_row.addStretch(1)
        hero_col.addLayout(status_row)
        hero_col.addSpacing(8)

        # Section eyebrow
        eyebrow = QLabel("DETECTED SIGN")
        eyebrow.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 10px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 3px; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        hero_col.addWidget(eyebrow)
        hero_col.addSpacing(2)

        # The big prediction word
        self._sign_lbl = QLabel("—")
        self._sign_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._sign_lbl.setMinimumHeight(110)
        self._sign_lbl.setStyleSheet(self._sign_qss(TEXT_HINT))
        self._sign_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred,
        )
        hero_col.addWidget(self._sign_lbl)

        # Confidence row: animated bar + percentage
        conf_row = QHBoxLayout()
        conf_row.setSpacing(14)
        conf_row.setContentsMargins(0, 4, 0, 0)

        self._conf_bar = AnimatedConfidenceBar(height=6)
        self._conf_bar.setMinimumWidth(280)
        conf_row.addWidget(self._conf_bar, 1)

        self._conf_pct_lbl = QLabel("—")
        self._conf_pct_lbl.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 13px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-weight: 700; letter-spacing: 1px;"
            f" background: transparent; border: none;"
        )
        self._conf_pct_lbl.setFixedWidth(64)
        self._conf_pct_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        conf_row.addWidget(self._conf_pct_lbl)

        hero_col.addLayout(conf_row)
        hero_col.addSpacing(22)

        # Phrase / sentence panel
        self._build_sentence_panel(hero_col)

        hero_col.addStretch(1)
        content_lay.addLayout(hero_col, 1)

        # ── Right column: camera + telemetry ─────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(14)
        right_col.setContentsMargins(0, 0, 0, 0)

        cam_panel = GlassCard(radius=8, bg=BG_BASE, border_color=HAIR)
        cam_panel.setFixedWidth(_CAM_W + 28)
        cam_panel_lay = QVBoxLayout(cam_panel)
        cam_panel_lay.setContentsMargins(14, 14, 14, 14)
        cam_panel_lay.setSpacing(10)

        cam_header = QHBoxLayout()
        cam_header.setContentsMargins(0, 0, 0, 0)
        cam_header.setSpacing(8)
        cam_header.addWidget(SectionLabel("LIVE FEED"))
        cam_header.addStretch(1)
        self._frame_lbl = QLabel("0000")
        self._frame_lbl.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 10px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 1.5px; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        cam_header.addWidget(self._frame_lbl)
        cam_panel_lay.addLayout(cam_header)

        self._cam_widget = CameraFrame(_CAM_W, _CAM_H, radius=6)
        cam_panel_lay.addWidget(self._cam_widget, 0, Qt.AlignmentFlag.AlignCenter)
        self._cam_card = cam_panel

        # Stat strip — three columns
        stat_strip = QFrame()
        stat_strip.setFixedHeight(56)
        stat_strip.setStyleSheet(
            f"QFrame {{ background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR}; border-radius: 6px; }}"
            f"QLabel {{ background: transparent; border: none; }}"
        )
        stat_lay = QHBoxLayout(stat_strip)
        stat_lay.setContentsMargins(0, 0, 0, 0)
        stat_lay.setSpacing(0)

        self._stat_lh   = self._make_stat_cell(stat_lay, "L HAND")
        self._stat_pose = self._make_stat_cell(stat_lay, "POSE")
        self._stat_rh   = self._make_stat_cell(stat_lay, "R HAND", last=True)

        cam_panel_lay.addWidget(stat_strip)

        # Mute pill row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        ctrl_row.addStretch(1)
        self._mute_btn = OutlineButton("MUTE TTS")
        self._mute_btn.setFixedWidth(140)
        self._mute_btn.clicked.connect(self._toggle_mute)
        ctrl_row.addWidget(self._mute_btn)
        cam_panel_lay.addLayout(ctrl_row)

        right_col.addWidget(cam_panel)

        # LSTM card (only if holistic available)
        if self._holistic:
            lstm_card = GlassCard(radius=6, bg=BG_SURFACE, border_color=HAIR)
            lstm_card.setFixedWidth(_CAM_W + 28)
            lstm_inner = QVBoxLayout(lstm_card)
            lstm_inner.setContentsMargins(18, 14, 18, 14)
            lstm_inner.setSpacing(8)

            lstm_hdr = QHBoxLayout()
            lstm_hdr.addWidget(SectionLabel("Dynamic · LSTM"))
            lstm_hdr.addStretch(1)
            self._lstm_status_dot = StatusDot(color=TEXT_HINT, size=8)
            lstm_hdr.addWidget(self._lstm_status_dot)
            lstm_inner.addLayout(lstm_hdr)

            self._seq_fill_bar = AnimatedConfidenceBar(height=4)
            lstm_inner.addWidget(self._seq_fill_bar)

            lstm_row = QHBoxLayout()
            lstm_row.setContentsMargins(0, 4, 0, 0)
            self._lstm_sign_lbl = QLabel("—")
            self._lstm_sign_lbl.setStyleSheet(
                f"font-size: 22px; color: {TEXT_HINT};"
                f" font-family: '{FONT_DISPLAY}', sans-serif;"
                f" font-weight: 700; letter-spacing: 0.5px;"
                f" background: transparent; border: none;"
            )
            lstm_row.addWidget(self._lstm_sign_lbl, 1)
            self._lstm_conf_lbl = QLabel("")
            self._lstm_conf_lbl.setStyleSheet(
                f"font-size: 11px; color: {TEXT_SEC};"
                f" font-family: '{FONT_MONO}', monospace;"
                f" font-weight: 700;"
                f" background: transparent; border: none;"
            )
            lstm_row.addWidget(
                self._lstm_conf_lbl, 0, Qt.AlignmentFlag.AlignRight,
            )
            lstm_inner.addLayout(lstm_row)
            right_col.addWidget(lstm_card)
        else:
            self._seq_fill_bar    = None
            self._lstm_sign_lbl   = None
            self._lstm_conf_lbl   = None
            self._lstm_status_dot = None

        # Model info card
        info_card = GlassCard(radius=6, bg=BG_SURFACE, border_color=HAIR)
        info_card.setFixedWidth(_CAM_W + 28)
        info_lay = QVBoxLayout(info_card)
        info_lay.setContentsMargins(18, 14, 18, 14)
        info_lay.setSpacing(6)
        info_lay.addWidget(SectionLabel("Model"))
        self._model_bar_lbl = QLabel("—")
        self._model_bar_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px;"
            f" font-family: '{FONT_MONO}', monospace; letter-spacing: 0.5px;"
            f" background: transparent; border: none;"
        )
        info_lay.addWidget(self._model_bar_lbl)
        self._model_info_lbl = QLabel("")
        self._model_info_lbl.setWordWrap(True)
        self._model_info_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: '{FONT_MONO}', monospace; line-height: 1.6;"
            f" background: transparent; border: none;"
        )
        info_lay.addWidget(self._model_info_lbl)
        right_col.addWidget(info_card)

        right_col.addStretch(1)
        content_lay.addLayout(right_col, 0)

    def _sign_qss(self, color: str) -> str:
        return (
            f"font-size: 88px; color: {color};"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            f" font-weight: 800; letter-spacing: -2px;"
            f" background: transparent; border: none;"
        )

    # ── Stat cell builder ────────────────────────────────────────────────────
    def _make_stat_cell(self, parent_lay: QHBoxLayout,
                        label: str, last: bool = False):
        cell = QWidget()
        cell.setStyleSheet("background: transparent;")
        cell_lay = QVBoxLayout(cell)
        cell_lay.setContentsMargins(0, 8, 0, 8)
        cell_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cell_lay.setSpacing(4)

        num_lbl = QLabel("00")
        num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_lbl.setStyleSheet(
            f"font-size: 20px; color: {TEXT_TITLE};"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-weight: 700;"
            f" background: transparent; border: none;"
        )
        cell_lay.addWidget(num_lbl)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"font-size: 8px; color: {TEXT_HINT};"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 1.5px; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        cell_lay.addWidget(lbl)

        parent_lay.addWidget(cell, 1)

        if not last:
            div = QWidget()
            div.setFixedWidth(1)
            div.setStyleSheet(f"background-color: {HAIR}; border: none;")
            parent_lay.addWidget(div)

        return num_lbl

    def _build_sentence_panel(self, parent_lay):
        panel = GlassCard(radius=8, bg=BG_SURFACE, border_color=HAIR)
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(22, 18, 22, 18)
        panel_lay.setSpacing(10)

        # Header row
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(10)

        trans_lbl = SectionLabel("Translated Phrase")
        hdr_row.addWidget(trans_lbl)

        self._nlp_tag = QLabel("NLP CORRECTED")
        self._nlp_tag.setStyleSheet(
            f"color: {ACCENT}; font-size: 9px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 1.5px; font-weight: 700;"
            f" border: 1px solid rgba(0,224,198,0.45);"
            f" border-radius: 3px;"
            f" padding: 2px 7px; background: rgba(0,224,198,0.08);"
        )
        self._nlp_tag.setVisible(False)
        hdr_row.addWidget(self._nlp_tag)
        hdr_row.addStretch(1)

        self._auto_add_cb = QCheckBox("Auto-add (hold ~1.5s)")
        self._auto_add_cb.setChecked(True)
        hdr_row.addWidget(self._auto_add_cb)

        panel_lay.addLayout(hdr_row)

        self._raw_lbl = QLabel("(no words yet)")
        self._raw_lbl.setWordWrap(True)
        self._raw_lbl.setStyleSheet(
            f"font-size: 22px; color: {TEXT_HINT};"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            f" font-weight: 500; letter-spacing: 0.2px;"
            f" background: transparent; border: none;"
        )
        panel_lay.addWidget(self._raw_lbl)

        self._corrected_lbl = QLabel("")
        self._corrected_lbl.setWordWrap(True)
        self._corrected_lbl.setStyleSheet(
            f"font-size: 13px; font-style: italic; color: {ACCENT};"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            f" background: transparent; border: none;"
        )
        panel_lay.addWidget(self._corrected_lbl)

        # Hold progress
        hold_row = QHBoxLayout()
        hold_row.setContentsMargins(0, 4, 0, 0)
        hold_row.setSpacing(10)
        self._hold_lbl = QLabel("HOLD —")
        self._hold_lbl.setStyleSheet(
            f"font-size: 9px; color: {TEXT_HINT};"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 1.5px; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        self._hold_lbl.setMinimumWidth(180)
        hold_row.addWidget(self._hold_lbl)
        self._hold_bar = AnimatedConfidenceBar(height=4)
        hold_row.addWidget(self._hold_bar, 1)
        panel_lay.addLayout(hold_row)

        # Action button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 6, 0, 0)

        add_btn = PillButton("+ ADD WORD", color=ACCENT)
        add_btn.clicked.connect(self._add_word_manual)
        btn_row.addWidget(add_btn)

        undo_btn = OutlineButton("UNDO")
        undo_btn.clicked.connect(self._undo_word)
        btn_row.addWidget(undo_btn)

        copy_btn = OutlineButton("COPY")
        copy_btn.clicked.connect(self._copy_sentence)
        btn_row.addWidget(copy_btn)

        speak_btn = OutlineButton("SPEAK")
        speak_btn.clicked.connect(self._speak_sentence)
        btn_row.addWidget(speak_btn)

        btn_row.addStretch(1)

        clear_btn = DangerButton("CLEAR")
        clear_btn.clicked.connect(self._clear_sentence)
        btn_row.addWidget(clear_btn)

        panel_lay.addLayout(btn_row)

        parent_lay.addWidget(panel)

    # ── Public API ───────────────────────────────────────────────────────────
    def activate(self):
        print("[PredictionScreen] Activated")
        self._start_camera()

    def deactivate(self):
        print("[PredictionScreen] Deactivated")
        self._stop_camera()
        self._reset_state()

    def cleanup(self):
        self._stop_camera()
        if self._owns_resources:
            try:
                self._detector.close()
            except Exception:
                pass

    def _reset_state(self):
        self._stable_count       = 0
        self._stable_sign        = None
        self._cooldown           = 0
        self._current_prediction = None
        self._seq_collector.clear()
        try:
            self._smoother.reset()
        except Exception:
            pass
        if self._speaker:
            try:
                self._speaker.reset()
            except Exception:
                pass

    # ── Camera (worker-based) ────────────────────────────────────────────────
    def _start_camera(self):
        ok = self._camera.start()
        if not ok:
            self._update_status("Camera not found", DANGER)
            print("[PredictionScreen] Camera not found")
            return

        self._update_status("Camera starting", TEXT_SEC)
        self._update_model_info()

        self._worker = CameraWorker(
            camera=self._camera,
            detector=self._detector,
            holistic=self._holistic,
            pose_detector=self._pose_detector,
            parent=self,
        )
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.start_capture()
        print("[PredictionScreen] Camera worker started")

    def _stop_camera(self):
        if self._worker is not None:
            self._worker.stop_capture()
            self._worker = None
        self._camera.stop()
        print("[PredictionScreen] Camera stopped")

    # ── Frame handler (main thread — called via queued signal) ───────────────
    def _on_frame_ready(self, annotated_frame, hand_lms_list,
                         pose_lms, face_lms, holistic_feats):
        self._frame_count += 1
        self._frame_lbl.setText(f"#{self._frame_count:04d}")

        lh_count   = len(hand_lms_list[0]) if len(hand_lms_list) > 0 else 0
        pose_count = len(pose_lms) if pose_lms is not None else 0
        rh_count   = len(hand_lms_list[1]) if len(hand_lms_list) > 1 else 0
        self._stat_lh.setText(f"{lh_count:02d}")
        self._stat_pose.setText(f"{pose_count:02d}")
        self._stat_rh.setText(f"{rh_count:02d}")

        frame = annotated_frame.copy()

        if hand_lms_list and self.model:
            self._cam_card.set_border_color(
                lerp(BG_BORDER, ACCENT, 0.35),
            )

            rf = self.model.model
            legacy = (
                rf is not None
                and getattr(rf, "n_features_in_", None) == 63
            )
            if legacy:
                features = self._detector.extract_features(hand_lms_list[0])
            else:
                features = self._spatial_extractor.extract_from_holistic(
                    hand_lms_list, pose_lms, face_lms,
                )

            raw_sign, raw_conf = self.model.predict(features)

            self._smoother.update(raw_sign, raw_conf or 0.0)
            sm_sign, sm_conf = self._smoother.best()
            sign_name  = sm_sign if sm_sign else raw_sign
            confidence = sm_conf if sm_sign else (raw_conf or 0.0)

            if pose_lms is not None:
                self._spatial_extractor.draw_debug(
                    frame, features, pose_lms,
                    face_lms=face_lms, hand_lms_list=hand_lms_list,
                )

            if sign_name and confidence >= 0.80:
                self._show_prediction(sign_name, confidence, tier="high")
            elif sign_name and confidence >= self._confidence_threshold:
                self._show_prediction(sign_name, confidence, tier="med")
            else:
                self._show_low_confidence(confidence if confidence else 0.0)

            me_fired = False
            if pose_lms is not None:
                me_fired = self._check_and_draw_me(
                    frame, hand_lms_list, pose_lms,
                )
            if not me_fired and sign_name:
                self._draw_hud(frame, sign_name, confidence)

            self._draw_contact_indicator(frame, features)
        else:
            self._cam_card.set_border_color(HAIR)
            self._show_no_hand()

        # LSTM rolling-window inference
        if self._holistic and holistic_feats is not None:
            self._seq_collector.add(holistic_feats)
            fill_ratio = self._seq_collector.fill_ratio()
            self._update_lstm_seq_bar(fill_ratio)
            if (self._seq_collector.is_ready()
                    and self._lstm_model
                    and self._lstm_model.is_trained()):
                seq  = self._seq_collector.get_sequence()
                lstm_sign, lstm_conf = self._lstm_model.predict(seq)
                self._show_lstm_prediction(lstm_sign, lstm_conf)

        self._cam_widget.set_frame(cv2_to_qpixmap(frame))

    # ── Display helpers ──────────────────────────────────────────────────────
    def _update_status(self, text: str, color: str):
        self._status_lbl.setText(text.upper())
        self._status_lbl.setStyleSheet(
            f"color: {color}; font-size: 11px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 2px; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        # PulsingDot — recreate to swap colour cleanly
        try:
            self._status_dot_anim._on  = color
            self._status_dot_anim._off = lerp(color, BG_DEEP, 0.7)
            self._status_dot_anim._update()
        except Exception:
            pass

    def _show_prediction(self, sign_name: str, confidence: float,
                          tier: str = "high"):
        display = sign_name.replace("_", " ").upper()
        pct     = int(round(confidence * 100))
        color   = confidence_color(confidence)
        suffix  = "" if tier == "high" else "?"

        self._sign_lbl.setText(f"{display}{suffix}")
        self._sign_lbl.setStyleSheet(self._sign_qss(color))
        self._conf_pct_lbl.setText(f"{pct}%")
        self._conf_pct_lbl.setStyleSheet(
            f"color: {color}; font-size: 13px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-weight: 700; letter-spacing: 1px;"
            f" background: transparent; border: none;"
        )
        self._conf_bar.set_confidence(confidence)
        self._update_status(f"{display} · {pct}%", color)
        self._current_prediction = sign_name

        if self._speaker and confidence >= self._confidence_threshold:
            self._speaker.say(display)
        self._check_auto_add(sign_name)

    def _show_low_confidence(self, confidence: float):
        color = confidence_color(confidence)
        self._sign_lbl.setText("?")
        self._sign_lbl.setStyleSheet(self._sign_qss(WARNING))
        pct = int(round(confidence * 100))
        self._conf_pct_lbl.setText(f"{pct}%")
        self._conf_pct_lbl.setStyleSheet(
            f"color: {WARNING}; font-size: 13px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-weight: 700; letter-spacing: 1px;"
            f" background: transparent; border: none;"
        )
        self._conf_bar.set_confidence(confidence)
        self._update_status("Uncertain · reposition hand", WARNING)
        self._stable_count = 0
        self._stable_sign  = None
        self._update_hold_progress(0.0, None)

    def _show_no_hand(self):
        self._sign_lbl.setText("—")
        self._sign_lbl.setStyleSheet(self._sign_qss(TEXT_HINT))
        self._conf_pct_lbl.setText("—")
        self._conf_pct_lbl.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 13px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-weight: 700; letter-spacing: 1px;"
            f" background: transparent; border: none;"
        )
        self._conf_bar.set_confidence(0.0)
        self._update_status("No hand detected", TEXT_SEC)
        self._stable_count       = 0
        self._stable_sign        = None
        self._current_prediction = None
        self._update_hold_progress(0.0, None)

    def _update_lstm_seq_bar(self, ratio: float):
        if self._seq_fill_bar is None:
            return
        self._seq_fill_bar.set_confidence(max(0.0, min(1.0, ratio)))

    def _show_lstm_prediction(self, sign: str, confidence: float):
        if self._lstm_sign_lbl is None:
            return
        if sign and confidence >= _LSTM_THRESHOLD:
            display = sign.replace("_", " ").upper()
            color   = ACCENT if confidence >= 0.80 else WARNING
            self._lstm_sign_lbl.setText(display)
            self._lstm_sign_lbl.setStyleSheet(
                f"font-size: 22px; color: {color};"
                f" font-family: '{FONT_DISPLAY}', sans-serif;"
                f" font-weight: 700; letter-spacing: 0.5px;"
                f" background: transparent; border: none;"
            )
            self._lstm_conf_lbl.setText(f"{int(confidence * 100)}%")
            if self._lstm_status_dot:
                self._lstm_status_dot.set_color(color)
        else:
            self._lstm_sign_lbl.setText("—")
            self._lstm_sign_lbl.setStyleSheet(
                f"font-size: 22px; color: {TEXT_HINT};"
                f" font-family: '{FONT_DISPLAY}', sans-serif;"
                f" font-weight: 700; letter-spacing: 0.5px;"
                f" background: transparent; border: none;"
            )
            self._lstm_conf_lbl.setText("")
            if self._lstm_status_dot:
                self._lstm_status_dot.set_color(TEXT_HINT)

    # ── cv2 overlay helpers ──────────────────────────────────────────────────
    def _check_and_draw_me(self, frame, hand_lms_list, pose_lms) -> bool:
        try:
            if not self.model.is_me_sign(hand_lms_list, pose_lms):
                return False
        except Exception:
            return False
        h, w = frame.shape[:2]
        tip  = hand_lms_list[0][8]
        tx, ty = int(tip.x * w), int(tip.y * h)
        cv2.circle(frame, (tx, ty), 18, (198, 224, 0), 2)
        cv2.putText(
            frame, "ME!", (tx + 22, ty + 6),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, (198, 224, 0), 2, cv2.LINE_AA,
        )
        self._draw_hud(frame, "ME", 1.0)
        return True

    def _draw_hud(self, frame, sign_name: str, confidence: float):
        if not sign_name:
            return
        panel_w, panel_h = 220, 64
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (8, 8), (8 + panel_w, 8 + panel_h),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        color = (198, 224, 0) if confidence >= 0.80 else (77, 178, 255)
        label = sign_name.replace("_", " ").upper()
        cv2.putText(
            frame, label, (16, 38),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"{int(confidence * 100)}%", (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )

    def _draw_contact_indicator(self, frame: np.ndarray,
                                features: np.ndarray) -> None:
        if len(features) < 65:
            return
        left_touching  = any(features[52 + 3 : 52 + 6] > 0.5)
        right_touching = any(features[58 + 3 : 58 + 6] > 0.5)
        if not (left_touching or right_touching):
            return
        h, w = frame.shape[:2]
        cx, cy = w - 22, 22
        cv2.circle(frame, (cx, cy), 8, (119, 85, 255), -1)
        cv2.putText(
            frame, "touching",
            (cx - 56, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (119, 85, 255), 1, cv2.LINE_AA,
        )

    # ── Auto-add logic ───────────────────────────────────────────────────────
    def _check_auto_add(self, sign_name: str):
        if not self._auto_add_cb.isChecked():
            self._stable_count = 0
            self._stable_sign  = None
            self._update_hold_progress(0.0, None)
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            self._update_hold_progress(0.0, None)
            return
        if sign_name == self._stable_sign:
            self._stable_count += 1
        else:
            self._stable_sign  = sign_name
            self._stable_count = 1
        ratio = min(1.0, self._stable_count / AUTO_ADD_FRAMES)
        self._update_hold_progress(ratio, sign_name)
        if self._stable_count >= AUTO_ADD_FRAMES:
            if self._sentence_buffer.add_word(sign_name):
                self._maybe_apply_phrase_match()
                self._update_sentence_display()
            self._stable_count = 0
            self._stable_sign  = None
            self._cooldown     = COOLDOWN_FRAMES

    def _update_hold_progress(self, ratio: float, sign_name):
        if sign_name and ratio > 0:
            display = sign_name.replace("_", " ").upper()
            self._hold_lbl.setText(f"HOLD {display} · {int(ratio * 100)}%")
            self._hold_lbl.setStyleSheet(
                f"font-size: 9px; color: {TEXT_PRIMARY};"
                f" font-family: '{FONT_MONO}', monospace;"
                f" letter-spacing: 1.5px; font-weight: 700;"
                f" background: transparent; border: none;"
            )
        else:
            self._hold_lbl.setText("HOLD —")
            self._hold_lbl.setStyleSheet(
                f"font-size: 9px; color: {TEXT_HINT};"
                f" font-family: '{FONT_MONO}', monospace;"
                f" letter-spacing: 1.5px; font-weight: 600;"
                f" background: transparent; border: none;"
            )
        self._hold_bar.set_confidence(ratio)

    # ── Sentence builder callbacks ───────────────────────────────────────────
    def _add_word_manual(self):
        if self._current_prediction:
            if self._sentence_buffer.add_word(self._current_prediction):
                self._stable_count = 0
                self._cooldown     = COOLDOWN_FRAMES
                self._maybe_apply_phrase_match()
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

    def _copy_sentence(self):
        words = self._sentence_buffer.get_words()
        if words:
            corrected = self._nlp.process(words)
            text = corrected if corrected else " ".join(words)
            QApplication.clipboard().setText(text)

    def _speak_sentence(self):
        if self._speaker and not self._sentence_buffer.is_empty():
            words     = self._sentence_buffer.get_words()
            corrected = self._nlp.process(words)
            if corrected:
                self._speaker.speak_sentence(corrected)

    def _update_sentence_display(self):
        words = self._sentence_buffer.get_words()
        if words:
            self._raw_lbl.setText(" ".join(w.replace("_", " ") for w in words))
            self._raw_lbl.setStyleSheet(
                f"font-size: 22px; color: {TEXT_TITLE};"
                f" font-family: '{FONT_DISPLAY}', sans-serif;"
                f" font-weight: 600; letter-spacing: 0.2px;"
                f" background: transparent; border: none;"
            )
            corrected = self._nlp.process(words)
            self._corrected_lbl.setText(corrected)
            raw_joined = " ".join(w.replace("_", " ") for w in words)
            self._nlp_tag.setVisible(
                bool(corrected) and corrected.lower() != raw_joined.lower()
            )
        else:
            self._raw_lbl.setText("(no words yet)")
            self._raw_lbl.setStyleSheet(
                f"font-size: 22px; color: {TEXT_HINT};"
                f" font-family: '{FONT_DISPLAY}', sans-serif;"
                f" font-weight: 500; letter-spacing: 0.2px;"
                f" background: transparent; border: none;"
            )
            self._corrected_lbl.setText("")
            self._nlp_tag.setVisible(False)

    def _update_model_info(self):
        if not self.model:
            return
        signs = self.model.get_all_signs()
        if signs:
            self._model_bar_lbl.setText(
                f"Random Forest · {len(signs)} classes"
            )
            self._model_info_lbl.setText(
                "  ".join(s.replace("_", " ").upper() for s in signs)
            )
        else:
            self._model_bar_lbl.setText("No model — train a sign first")
            self._model_info_lbl.setText("")

    # ── Phrase matching ──────────────────────────────────────────────────────
    def _maybe_apply_phrase_match(self):
        if self._phrase_matcher is None:
            return
        words = self._sentence_buffer.get_words()
        try:
            result = self._phrase_matcher.match_tail(words)
        except Exception as exc:
            print(f"[PredictionScreen] phrase match failed: {exc}")
            return
        if not result:
            return
        matched_count, output = result
        if matched_count <= 0 or not output:
            return
        for _ in range(matched_count):
            self._sentence_buffer.undo()
        self._sentence_buffer.add_word(output)
        if self._speaker:
            try:
                self._speaker.speak_sentence(output)
            except Exception:
                pass

    # ── Settings + dependency injection (called by MainWindow) ───────────────
    def apply_settings(self, settings: dict) -> None:
        self._confidence_threshold = float(
            settings.get("confidence_threshold", 0.6)
        )
        new_window = max(1, int(settings.get("smoothing_window", 5)))
        if new_window != self._smoothing_window:
            self._smoothing_window = new_window
            self._smoother = TemporalSmoother(
                window=self._smoothing_window, min_vote_share=0.55,
            )
        if self._speaker:
            try:
                self._speaker.set_enabled(
                    bool(settings.get("tts_enabled", True))
                )
                self._speaker.set_voice(
                    settings.get("tts_voice", "") or ""
                )
            except Exception as exc:
                print(f"[PredictionScreen] TTS settings update failed: {exc}")

    def set_phrase_matcher(self, matcher) -> None:
        self._phrase_matcher = matcher

    def set_camera(self, camera) -> None:
        self._stop_camera()
        self._camera = camera

    def set_holistic(self, holistic) -> None:
        self._stop_camera()
        self._holistic = holistic

    def restart_camera(self) -> None:
        self._stop_camera()
        self._start_camera()

    # ── TTS toggle ───────────────────────────────────────────────────────────
    def _toggle_mute(self):
        self._tts_muted = not self._tts_muted
        if self._speaker:
            self._speaker.set_enabled(not self._tts_muted)
        self._mute_btn.setText("UNMUTE TTS" if self._tts_muted else "MUTE TTS")

    def _on_back(self):
        self._stop_camera()
        if self.on_back:
            self.on_back()
