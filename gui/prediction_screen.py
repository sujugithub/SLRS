"""
Prediction screen — PyQt6.
Camera capture + MediaPipe detection run in CameraWorker (QThread).
The main thread receives annotated frames and detection data via signal,
then runs the lightweight RF / LSTM inference and updates the UI.
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
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
    HAIR,
    CameraFrame, GlassCard, HSep, PillButton, DangerButton, OutlineButton,
    SectionLabel, StatusDot, cv2_to_qpixmap, lerp,
)

AUTO_ADD_FRAMES      = 10
COOLDOWN_FRAMES      = 60
_LSTM_THRESHOLD      = 0.55

_CAM_W = 580
_CAM_H = 435


class PredictionScreen(QWidget):
    """Screen for real-time sign language prediction with live camera."""

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

        # Settings-driven knobs (overwritten by apply_settings).
        self._confidence_threshold = 0.6
        self._smoothing_window     = 5
        self._smoother             = TemporalSmoother(
            window=self._smoothing_window, min_vote_share=0.55,
        )
        self._phrase_matcher = None  # set via set_phrase_matcher()

        # Camera worker (replaces QTimer camera loop).
        self._worker: CameraWorker | None = None

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Nav bar ───────────────────────────────────────────────────────────
        nav = QWidget()
        nav.setFixedHeight(48)
        nav.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-bottom: 1px solid {HAIR};")
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(20, 0, 20, 0)
        nav_lay.setSpacing(12)

        title_lbl = QLabel("Live Detection")
        title_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {TEXT_TITLE};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;")
        nav_lay.addWidget(title_lbl, 1)

        self._hand_status_nav = QLabel("")
        self._hand_status_nav.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        nav_lay.addWidget(self._hand_status_nav, 0, Qt.AlignmentFlag.AlignRight)

        self._mute_btn = PillButton("MUTE", color=BG_ELEVATED,
                                    hover_color=BG_FLOAT, text_color=TEXT_SEC)
        self._mute_btn.setFixedWidth(90)
        self._mute_btn.clicked.connect(self._toggle_mute)
        nav_lay.addWidget(self._mute_btn)

        root.addWidget(nav)

        # ── Main content ──────────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet(f"background-color: {BG_DEEP};")
        content_lay = QHBoxLayout(content)
        content_lay.setContentsMargins(20, 14, 20, 14)
        content_lay.setSpacing(16)
        root.addWidget(content, 1)

        # ── Left: camera + stat strip + model info ────────────────────────────
        left_lay = QVBoxLayout()
        left_lay.setSpacing(0)

        # Camera panel
        cam_panel = QWidget()
        cam_panel.setStyleSheet(
            f"background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR};")
        cam_panel_lay = QVBoxLayout(cam_panel)
        cam_panel_lay.setContentsMargins(0, 0, 0, 0)
        cam_panel_lay.setSpacing(0)
        self._cam_widget = CameraFrame(_CAM_W, _CAM_H, radius=0)
        cam_panel_lay.addWidget(self._cam_widget)
        self._cam_card = cam_panel   # kept as _cam_card for border-color updates
        left_lay.addWidget(cam_panel)

        # Landmark stat strip — 3 columns separated by 1px hairlines
        stat_strip = QWidget()
        stat_strip.setFixedHeight(56)
        stat_strip.setStyleSheet(
            f"background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR}; border-top: none;")
        stat_lay = QHBoxLayout(stat_strip)
        stat_lay.setContentsMargins(0, 0, 0, 0)
        stat_lay.setSpacing(0)

        self._stat_lh   = self._make_stat_cell(stat_lay, "L HAND LMS")
        self._stat_pose = self._make_stat_cell(stat_lay, "POSE LMS")
        self._stat_rh   = self._make_stat_cell(stat_lay, "R HAND LMS", last=True)

        left_lay.addWidget(stat_strip)

        self._model_info_lbl = QLabel("")
        self._model_info_lbl.setWordWrap(True)
        self._model_info_lbl.setContentsMargins(0, 6, 0, 0)
        self._model_info_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        left_lay.addWidget(self._model_info_lbl)
        left_lay.addStretch(1)
        content_lay.addLayout(left_lay, 3)

        # ── Right panel ───────────────────────────────────────────────────────
        right_lay = QVBoxLayout()
        right_lay.setSpacing(8)
        right_lay.setContentsMargins(0, 0, 0, 0)

        # Sign + confidence card (merged)
        sign_card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        sign_card.setFixedWidth(260)
        sign_inner = QVBoxLayout(sign_card)
        sign_inner.setContentsMargins(16, 14, 16, 14)
        sign_inner.setSpacing(8)
        sign_inner.addWidget(SectionLabel("Detected Sign"))

        self._sign_lbl = QLabel("---")
        self._sign_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sign_lbl.setMinimumHeight(72)
        self._sign_lbl.setStyleSheet(
            f"font-size: 48px; color: {TEXT_HINT};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;")
        sign_inner.addWidget(self._sign_lbl)

        # 2px slim confidence bar
        self._conf_bar = QProgressBar()
        self._conf_bar.setFixedHeight(2)
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {HAIR};"
            f" height: 2px; border-radius: 0; }}"
            f"QProgressBar::chunk {{ background: {SUCCESS}; border-radius: 0; }}")
        sign_inner.addWidget(self._conf_bar)

        # Confidence % label (small, below bar)
        self._conf_pct_lbl = QLabel("---%")
        self._conf_pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._conf_pct_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        sign_inner.addWidget(self._conf_pct_lbl)
        right_lay.addWidget(sign_card)

        # Status card
        status_card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        status_card.setFixedWidth(260)
        status_inner = QVBoxLayout(status_card)
        status_inner.setContentsMargins(16, 12, 16, 12)
        status_inner.setSpacing(6)
        status_inner.addWidget(SectionLabel("Status"))

        status_row = QHBoxLayout()
        self._status_dot = StatusDot(color=TEXT_HINT, size=7)
        status_row.addWidget(self._status_dot)
        status_row.addSpacing(6)
        self._status_lbl = QLabel("Initializing…")
        self._status_lbl.setStyleSheet(
            f"font-size: 11px; color: {TEXT_SEC};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        status_row.addWidget(self._status_lbl, 1)
        status_inner.addLayout(status_row)
        right_lay.addWidget(status_card)

        # LSTM card (only if holistic available)
        if self._holistic:
            lstm_card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
            lstm_card.setFixedWidth(260)
            lstm_inner = QVBoxLayout(lstm_card)
            lstm_inner.setContentsMargins(16, 12, 16, 12)
            lstm_inner.setSpacing(6)

            lstm_hdr = QHBoxLayout()
            lstm_hdr.addWidget(SectionLabel("Dynamic (LSTM)"))
            lstm_hdr.addStretch(1)
            self._lstm_status_dot = StatusDot(color=TEXT_HINT, size=7)
            lstm_hdr.addWidget(self._lstm_status_dot)
            lstm_inner.addLayout(lstm_hdr)

            self._seq_fill_bar = QProgressBar()
            self._seq_fill_bar.setFixedHeight(2)
            self._seq_fill_bar.setRange(0, 100)
            self._seq_fill_bar.setValue(0)
            self._seq_fill_bar.setTextVisible(False)
            self._seq_fill_bar.setStyleSheet(
                f"QProgressBar {{ border: none; background: {HAIR};"
                f" height: 2px; border-radius: 0; }}"
                f"QProgressBar::chunk {{ background: {ACCENT}; border-radius: 0; }}")
            lstm_inner.addWidget(self._seq_fill_bar)

            self._lstm_sign_lbl = QLabel("---")
            self._lstm_sign_lbl.setStyleSheet(
                f"font-size: 20px; color: {TEXT_HINT};"
                f" font-family: Georgia, 'Times New Roman', serif;"
                " background: transparent; border: none;")
            lstm_inner.addWidget(self._lstm_sign_lbl)

            self._lstm_conf_lbl = QLabel("")
            self._lstm_conf_lbl.setStyleSheet(
                f"font-size: 10px; color: {TEXT_SEC};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;")
            lstm_inner.addWidget(self._lstm_conf_lbl)
            right_lay.addWidget(lstm_card)
        else:
            self._seq_fill_bar    = None
            self._lstm_sign_lbl   = None
            self._lstm_conf_lbl   = None
            self._lstm_status_dot = None

        right_lay.addStretch(1)
        content_lay.addLayout(right_lay, 0)

        # ── Sentence builder panel ────────────────────────────────────────────
        self._build_sentence_panel(root)

        # ── Custom bottom status bar (32px) ───────────────────────────────────
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(32)
        bottom_bar.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-top: 1px solid {HAIR};")
        bb_lay = QHBoxLayout(bottom_bar)
        bb_lay.setContentsMargins(20, 0, 20, 0)
        bb_lay.setSpacing(0)

        self._model_bar_lbl = QLabel("—")
        self._model_bar_lbl.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 10px;"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        bb_lay.addWidget(self._model_bar_lbl)
        bb_lay.addStretch(1)

        self._frame_lbl = QLabel("Frame: 000")
        self._frame_lbl.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 10px;"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        bb_lay.addWidget(self._frame_lbl)

        root.addWidget(bottom_bar)

    # ── Stat cell builder ─────────────────────────────────────────────────────
    def _make_stat_cell(self, parent_lay: QHBoxLayout,
                        label: str, last: bool = False):
        """Build one column of the 3-column stat strip, return the number QLabel."""
        cell = QWidget()
        cell.setStyleSheet("background: transparent;")
        cell_lay = QVBoxLayout(cell)
        cell_lay.setContentsMargins(0, 8, 0, 8)
        cell_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cell_lay.setSpacing(2)

        num_lbl = QLabel("00")
        num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_lbl.setStyleSheet(
            f"font-size: 22px; color: {TEXT_PRIMARY};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;")
        cell_lay.addWidget(num_lbl)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"font-size: 8px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        cell_lay.addWidget(lbl)

        parent_lay.addWidget(cell, 1)

        if not last:
            div = QWidget()
            div.setFixedWidth(1)
            div.setStyleSheet(f"background-color: {HAIR}; border: none;")
            parent_lay.addWidget(div)

        return num_lbl

    def _build_sentence_panel(self, root_layout):
        panel = QWidget()
        panel.setStyleSheet(
            f"background-color: {BG_SURFACE};"
            f" border-top: 1px solid {HAIR};")
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(20, 12, 20, 12)
        panel_lay.setSpacing(8)

        # Header row
        hdr_row = QHBoxLayout()

        trans_lbl = QLabel("TRANSLATED SENTENCE")
        trans_lbl.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 9px;"
            f" font-family: 'Courier New', monospace; letter-spacing: 2px;"
            " background: transparent; border: none;")
        hdr_row.addWidget(trans_lbl)
        hdr_row.addSpacing(12)

        # "NLP corrected" tag — hidden until NLP produces a different result
        self._nlp_tag = QLabel("NLP CORRECTED")
        self._nlp_tag.setStyleSheet(
            f"color: {SUCCESS}; font-size: 10px;"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            f" border: 1px solid rgba(123,176,138,0.35);"
            f" padding: 1px 6px; background: transparent;")
        self._nlp_tag.setVisible(False)
        hdr_row.addWidget(self._nlp_tag)
        hdr_row.addStretch(1)

        self._auto_add_cb = QCheckBox("AUTO-ADD  (hold ~1.5 s)")
        self._auto_add_cb.setChecked(True)
        hdr_row.addWidget(self._auto_add_cb)
        panel_lay.addLayout(hdr_row)

        # Sentence text area
        sentence_area = QWidget()
        sentence_area.setStyleSheet(
            f"background-color: {BG_DEEP};"
            f" border: 1px solid {HAIR};")
        sa_lay = QVBoxLayout(sentence_area)
        sa_lay.setContentsMargins(12, 8, 12, 8)
        sa_lay.setSpacing(4)

        self._raw_lbl = QLabel("(empty — hold a gesture or press Add Word)")
        self._raw_lbl.setWordWrap(True)
        self._raw_lbl.setStyleSheet(
            f"font-size: 14px; color: {TEXT_SEC};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        sa_lay.addWidget(self._raw_lbl)

        self._corrected_lbl = QLabel("")
        self._corrected_lbl.setWordWrap(True)
        self._corrected_lbl.setStyleSheet(
            f"font-size: 12px; font-style: italic; color: {SUCCESS};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        sa_lay.addWidget(self._corrected_lbl)
        panel_lay.addWidget(sentence_area)

        # Hold progress + buttons row
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        # Hold progress
        hold_col = QVBoxLayout()
        hold_col.setSpacing(3)
        self._hold_lbl = QLabel("HOLD: ---")
        self._hold_lbl.setStyleSheet(
            f"font-size: 9px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        hold_col.addWidget(self._hold_lbl)
        self._hold_bar = QProgressBar()
        self._hold_bar.setFixedHeight(2)
        self._hold_bar.setRange(0, 100)
        self._hold_bar.setValue(0)
        self._hold_bar.setTextVisible(False)
        self._hold_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {HAIR};"
            f" height: 2px; border-radius: 0; }}"
            f"QProgressBar::chunk {{ background: {ACCENT}; border-radius: 0; }}")
        hold_col.addWidget(self._hold_bar)
        hold_col.addStretch(1)
        bottom_row.addLayout(hold_col, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        add_btn = PillButton("+ ADD WORD", color=ACCENT, hover_color="#5a8c69")
        add_btn.clicked.connect(self._add_word_manual)
        btn_row.addWidget(add_btn)

        undo_btn = OutlineButton("↩ UNDO")
        undo_btn.clicked.connect(self._undo_word)
        btn_row.addWidget(undo_btn)

        copy_btn = OutlineButton("COPY")
        copy_btn.clicked.connect(self._copy_sentence)
        btn_row.addWidget(copy_btn)

        speak_btn = OutlineButton("SPEAK")
        speak_btn.clicked.connect(self._speak_sentence)
        btn_row.addWidget(speak_btn)

        clear_btn = DangerButton("CLEAR")
        clear_btn.clicked.connect(self._clear_sentence)
        btn_row.addWidget(clear_btn)

        bottom_row.addLayout(btn_row)
        panel_lay.addLayout(bottom_row)

        root_layout.addWidget(panel)

    # ── Public API ────────────────────────────────────────────────────────────
    def activate(self):
        # NOTE: animation handled by MainWindow._switch_to(); no fade_in here.
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

    # ── Camera (worker-based) ─────────────────────────────────────────────────
    def _start_camera(self):
        ok = self._camera.start()
        if not ok:
            self._update_status("Camera not found", DANGER)
            print("[PredictionScreen] Camera not found")
            return

        self._update_status("Camera starting…", TEXT_SEC)
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

    # ── Frame handler (main thread — called via queued signal) ────────────────
    def _on_frame_ready(self, annotated_frame, hand_lms_list,
                         pose_lms, face_lms, holistic_feats):
        """
        Receives processed frame + detection data from CameraWorker.
        Runs RF prediction (fast), draws HUD, updates all UI labels.
        """
        # Increment frame counter
        self._frame_count += 1
        self._frame_lbl.setText(f"Frame: {self._frame_count:04d}")

        # Update landmark stat strip
        lh_count   = len(hand_lms_list[0]) if len(hand_lms_list) > 0 else 0
        pose_count = len(pose_lms) if pose_lms is not None else 0
        rh_count   = len(hand_lms_list[1]) if len(hand_lms_list) > 1 else 0
        self._stat_lh.setText(str(lh_count) if lh_count else "00")
        self._stat_pose.setText(str(pose_count) if pose_count else "00")
        self._stat_rh.setText(str(rh_count) if rh_count else "00")

        # Work on a mutable copy for HUD overlay.
        frame = annotated_frame.copy()

        # ── Static RF prediction ──────────────────────────────────────────
        if hand_lms_list and self.model:
            self._cam_card.setStyleSheet(
                f"background-color: {BG_SURFACE};"
                f" border: 1px solid {lerp(BG_BORDER, SUCCESS, 0.4)};")

            # Use the extractor that matches what the loaded model was trained on.
            # n_features_in_ == 63 → legacy raw-landmark model still in use.
            # Any other width (e.g. 52) → spatial extractor (new model after retrain).
            rf = self.model.model
            legacy = (rf is not None
                      and getattr(rf, "n_features_in_", None) == 63)
            if legacy:
                features = self._detector.extract_features(hand_lms_list[0])
            else:
                features = self._spatial_extractor.extract_from_holistic(
                    hand_lms_list, pose_lms, face_lms)

            raw_sign, raw_conf = self.model.predict(features)

            # Feed raw prediction into the smoothing window. The smoother
            # returns a confidence-weighted majority vote over the last
            # _smoothing_window frames; setting window=1 collapses to a no-op.
            self._smoother.update(raw_sign, raw_conf or 0.0)
            sm_sign, sm_conf = self._smoother.best()
            sign_name = sm_sign if sm_sign else raw_sign
            confidence = sm_conf if sm_sign else (raw_conf or 0.0)

            # Debug spatial overlay (lines from nose to wrists, zone colours,
            # contact lines from fingertips to face anchors)
            if pose_lms is not None:
                self._spatial_extractor.draw_debug(
                    frame, features, pose_lms,
                    face_lms=face_lms, hand_lms_list=hand_lms_list)

            if sign_name and confidence >= 0.80:
                self._show_prediction(sign_name, confidence, tier="high")
            elif sign_name and confidence >= self._confidence_threshold:
                self._show_prediction(sign_name, confidence, tier="med")
            else:
                self._show_low_confidence(confidence if confidence else 0.0)

            me_fired = False
            if pose_lms is not None:
                me_fired = self._check_and_draw_me(frame, hand_lms_list, pose_lms)
            if not me_fired and sign_name:
                self._draw_hud(frame, sign_name, confidence)

            # Contact indicator: small red dot + label if any touching flag active
            self._draw_contact_indicator(frame, features)
        else:
            self._cam_card.setStyleSheet(
                f"background-color: {BG_SURFACE};"
                f" border: 1px solid {HAIR};")
            self._show_no_hand()

        # ── LSTM rolling-window inference ─────────────────────────────────
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

        # Update camera display with HUD already drawn.
        self._cam_widget.set_frame(cv2_to_qpixmap(frame))

    # ── Display helpers ───────────────────────────────────────────────────────
    def _update_status(self, text: str, color: str):
        self._hand_status_nav.setText(text)
        self._hand_status_nav.setStyleSheet(
            f"font-size: 10px; color: {color};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"font-size: 11px; color: {color};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        self._status_dot.set_color(color)

    def _show_prediction(self, sign_name: str, confidence: float,
                          tier: str = "high"):
        display = sign_name.replace("_", " ").upper()
        pct     = int(confidence * 100)
        color   = SUCCESS if tier == "high" else WARNING
        status  = (f"{display}  {pct}%" if tier == "high"
                   else f"{display}?  {pct}%")

        self._sign_lbl.setText(display)
        self._sign_lbl.setStyleSheet(
            f"font-size: 48px; color: {color};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;")
        self._conf_pct_lbl.setText(f"{confidence:.0%}")
        self._conf_pct_lbl.setStyleSheet(
            f"font-size: 10px; color: {color};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        self._conf_bar.setValue(pct)
        self._conf_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {HAIR};"
            f" height: 2px; border-radius: 0; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 0; }}")
        self._update_status(status, color)
        self._current_prediction = sign_name
        # Gate TTS on confidence threshold so low-confidence flicker
        # doesn't trigger noisy speech. _speaker.say() already dedupes.
        if self._speaker and confidence >= self._confidence_threshold:
            self._speaker.say(display)
        self._check_auto_add(sign_name)

    def _show_low_confidence(self, confidence: float):
        self._sign_lbl.setText("?")
        self._sign_lbl.setStyleSheet(
            f"font-size: 48px; color: {WARNING};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;")
        self._conf_pct_lbl.setText(f"{confidence:.0%}")
        self._conf_bar.setValue(int(confidence * 100))
        self._update_status("Uncertain — reposition hand", WARNING)
        self._stable_count = 0
        self._stable_sign  = None
        self._update_hold_progress(0.0, None)

    def _show_no_hand(self):
        self._sign_lbl.setText("---")
        self._sign_lbl.setStyleSheet(
            f"font-size: 48px; color: {TEXT_HINT};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;")
        self._conf_pct_lbl.setText("---%")
        self._conf_bar.setValue(0)
        self._update_status("Show your hand to the camera", TEXT_SEC)
        self._stable_count       = 0
        self._stable_sign        = None
        self._current_prediction = None
        self._update_hold_progress(0.0, None)

    def _update_lstm_seq_bar(self, ratio: float):
        if self._seq_fill_bar is None:
            return
        val   = int(max(0.0, min(1.0, ratio)) * 100)
        color = SUCCESS if ratio >= 1.0 else ACCENT
        self._seq_fill_bar.setValue(val)
        self._seq_fill_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {HAIR};"
            f" height: 2px; border-radius: 0; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 0; }}")

    def _show_lstm_prediction(self, sign: str, confidence: float):
        if self._lstm_sign_lbl is None:
            return
        if sign and confidence >= _LSTM_THRESHOLD:
            display = sign.replace("_", " ").upper()
            color   = SUCCESS if confidence >= 0.80 else ACCENT
            self._lstm_sign_lbl.setText(display)
            self._lstm_sign_lbl.setStyleSheet(
                f"font-size: 20px; color: {color};"
                f" font-family: Georgia, 'Times New Roman', serif;"
                " background: transparent; border: none;")
            self._lstm_conf_lbl.setText(f"{confidence:.0%}")
            if self._lstm_status_dot:
                self._lstm_status_dot.set_color(color)
        else:
            self._lstm_sign_lbl.setText("---")
            self._lstm_sign_lbl.setStyleSheet(
                f"font-size: 20px; color: {TEXT_HINT};"
                f" font-family: Georgia, 'Times New Roman', serif;"
                " background: transparent; border: none;")
            self._lstm_conf_lbl.setText("")
            if self._lstm_status_dot:
                self._lstm_status_dot.set_color(TEXT_HINT)

    # ── cv2 overlay helpers ───────────────────────────────────────────────────
    def _check_and_draw_me(self, frame, hand_lms_list, pose_lms) -> bool:
        try:
            if not self.model.is_me_sign(hand_lms_list, pose_lms):
                return False
        except Exception:
            return False
        h, w = frame.shape[:2]
        tip  = hand_lms_list[0][8]
        tx, ty = int(tip.x * w), int(tip.y * h)
        cv2.circle(frame, (tx, ty), 18, (0, 255, 0), 2)
        cv2.putText(frame, "ME!", (tx + 22, ty + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        self._draw_hud(frame, "ME", 1.0)
        return True

    def _draw_hud(self, frame, sign_name: str, confidence: float):
        if not sign_name:
            return
        panel_w, panel_h = 220, 64
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        color = (0, 200, 0) if confidence >= 0.80 else (0, 200, 255)
        label = sign_name.replace("_", " ").upper()
        cv2.putText(frame, label, (16, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{int(confidence * 100)}%", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
                    cv2.LINE_AA)

    def _draw_contact_indicator(self, frame: np.ndarray,
                                features: np.ndarray) -> None:
        """Draw a small red dot + 'touching' label if any contact flag is active."""
        # Group G contact flags start at index 52 (left) and 58 (right).
        # Flags are at offsets +3, +4, +5 within each group.
        # Only check if the feature vector includes Group G (length >= 65).
        if len(features) < 65:
            return
        left_touching  = any(features[52 + 3 : 52 + 6] > 0.5)
        right_touching = any(features[58 + 3 : 58 + 6] > 0.5)
        if not (left_touching or right_touching):
            return
        h, w = frame.shape[:2]
        cx, cy = w - 22, 22
        cv2.circle(frame, (cx, cy), 8, (0, 0, 220), -1)
        cv2.putText(
            frame, "touching",
            (cx - 56, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 220), 1, cv2.LINE_AA,
        )

    # ── Auto-add logic ────────────────────────────────────────────────────────
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
            self._hold_lbl.setText(f"HOLD: {display}  {int(ratio * 100)}%")
            self._hold_lbl.setStyleSheet(
                f"font-size: 9px; color: {TEXT_PRIMARY};"
                f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
                " background: transparent; border: none;")
        else:
            self._hold_lbl.setText("HOLD: ---")
            self._hold_lbl.setStyleSheet(
                f"font-size: 9px; color: {TEXT_HINT};"
                f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
                " background: transparent; border: none;")
        self._hold_bar.setValue(int(ratio * 100))
        color = SUCCESS if ratio >= 1.0 else ACCENT
        self._hold_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {HAIR};"
            f" height: 2px; border-radius: 0; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 0; }}")

    # ── Sentence builder callbacks ────────────────────────────────────────────
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
            self._raw_lbl.setText(" ".join(words))
            self._raw_lbl.setStyleSheet(
                f"font-size: 14px; color: {TEXT_PRIMARY};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;")
            corrected = self._nlp.process(words)
            self._corrected_lbl.setText(corrected)
            # Show NLP tag when corrected text differs from raw words
            raw_joined = " ".join(w.replace("_", " ") for w in words)
            self._nlp_tag.setVisible(
                bool(corrected) and corrected.lower() != raw_joined.lower())
        else:
            self._raw_lbl.setText(
                "(empty — hold a gesture or press Add Word)")
            self._raw_lbl.setStyleSheet(
                f"font-size: 14px; color: {TEXT_SEC};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;")
            self._corrected_lbl.setText("")
            self._nlp_tag.setVisible(False)

    def _update_model_info(self):
        if not self.model:
            return
        signs = self.model.get_all_signs()
        if signs:
            self._model_bar_lbl.setText(
                f"Random Forest  ·  {len(signs)} classes")
            self._model_info_lbl.setText(
                "  ".join(s.replace("_", " ") for s in signs))
            self._model_info_lbl.setStyleSheet(
                f"font-size: 10px; color: {TEXT_HINT};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;")
        else:
            self._model_bar_lbl.setText("No model — train a sign first")
            self._model_info_lbl.setText("")

    # ── Phrase matching ───────────────────────────────────────────────────────
    def _maybe_apply_phrase_match(self):
        """Check the buffer tail against phrases.json; collapse a match.

        When a phrase like ['NICE','TO','MEET','YOU'] matches the tail, the
        matched words are removed from the buffer and a single output token
        is added in their place. The full phrase is also spoken as one
        utterance via TTSSpeaker.speak_sentence().
        """
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
        """Apply a fresh settings dict (live updates from SettingsDialog)."""
        self._confidence_threshold = float(
            settings.get("confidence_threshold", 0.6))
        new_window = max(1, int(settings.get("smoothing_window", 5)))
        if new_window != self._smoothing_window:
            self._smoothing_window = new_window
            self._smoother = TemporalSmoother(
                window=self._smoothing_window, min_vote_share=0.55,
            )
        if self._speaker:
            try:
                self._speaker.set_enabled(bool(settings.get("tts_enabled", True)))
                self._speaker.set_voice(settings.get("tts_voice", "") or "")
            except Exception as exc:
                print(f"[PredictionScreen] TTS settings update failed: {exc}")

    def set_phrase_matcher(self, matcher) -> None:
        """Inject the PhraseMatcher used for tail matching."""
        self._phrase_matcher = matcher

    def set_camera(self, camera) -> None:
        """Replace the camera handle and rebuild the worker on next activate."""
        self._stop_camera()
        self._camera = camera

    def set_holistic(self, holistic) -> None:
        """Replace the holistic detector and rebuild the worker on next activate."""
        self._stop_camera()
        self._holistic = holistic

    def restart_camera(self) -> None:
        """Stop and restart the camera worker (used after camera index change)."""
        self._stop_camera()
        self._start_camera()

    # ── TTS toggle ────────────────────────────────────────────────────────────
    def _toggle_mute(self):
        self._tts_muted = not self._tts_muted
        if self._speaker:
            self._speaker.set_enabled(not self._tts_muted)
        self._mute_btn.setText("UNMUTE" if self._tts_muted else "MUTE")

    def _on_back(self):
        self._stop_camera()
        if self.on_back:
            self.on_back()
