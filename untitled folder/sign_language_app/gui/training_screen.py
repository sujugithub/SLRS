"""
Training screen — PyQt6.
Camera capture + MediaPipe detection run in CameraWorker (QThread) so the
main thread is never blocked.  The auto-capture and sequence-recording logic
remain on the main thread, triggered by the worker's frame_ready signal.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QScrollArea, QVBoxLayout, QWidget,
)

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, SAMPLES_PER_CLASS,
    SEQ_LENGTH, MIN_SEQUENCES_PER_SIGN,
)
from core.feature_extractor import SpatialFeatureExtractor
from core.hand_detector import HandDetector
from core.camera_worker import CameraWorker
from PyQt6.QtGui import QKeySequence, QShortcut
from gui.design import (
    ACCENT, BG_BASE, BG_BORDER, BG_DEEP, BG_ELEVATED, BG_SURFACE, BG_FLOAT,
    DANGER, SUCCESS, WARNING, TEXT_HINT, TEXT_PRIMARY, TEXT_SEC, TEXT_TITLE,
    HAIR,
    CameraFrame, GlassCard, HSep, PillButton, DangerButton, OutlineButton,
    SectionLabel, cv2_to_qpixmap, lerp,
)

# ── Local constants ──────────────────────────────────────────────────────────
MIN_IMAGES            = 10
MAX_IMAGES            = SAMPLES_PER_CLASS
THUMB_SIZE            = 60
AUTO_CAPTURE_INTERVAL = 300      # ms between auto-capture ticks
AUTO_CAPTURE_TARGET   = 30
MIN_SEQS              = MIN_SEQUENCES_PER_SIGN
MAX_SEQS              = 30

_CAM_W = 580
_CAM_H = 435

_PROG_SS = (
    f"QProgressBar {{ border: none; background: {HAIR};"
    f" height: 4px; border-radius: 0; }}"
    f"QProgressBar::chunk {{ background: {ACCENT}; border-radius: 0; }}"
)
_PROG_SS_OK = (
    f"QProgressBar {{ border: none; background: {HAIR};"
    f" height: 4px; border-radius: 0; }}"
    f"QProgressBar::chunk {{ background: {SUCCESS}; border-radius: 0; }}"
)


class TrainingScreen(QWidget):
    """Screen for recording static frames and dynamic gesture sequences."""

    def __init__(self, on_back=None, on_save=None, on_save_sequences=None,
                 camera=None, detector=None, pose_detector=None,
                 holistic=None, parent=None):
        super().__init__(parent)
        self.on_back           = on_back
        self.on_save           = on_save
        self.on_save_sequences = on_save_sequences

        from core.camera_handler import CameraHandler
        self._camera         = camera   or CameraHandler()
        self._detector       = detector or HandDetector()
        self._pose_detector  = pose_detector
        self._owns_resources = camera is None

        self._holistic      = holistic
        self._owns_holistic = False

        # ── Spatial feature extractor ─────────────────────────────────────
        self._spatial_extractor  = SpatialFeatureExtractor()
        self._shape_printed      = False   # print feature shape once per session

        # ── State ────────────────────────────────────────────────────────────
        self._mode               = "static"
        self._contact_mode       = False   # toggled by 'C' key
        self.captured_features   = []
        self.captured_seqs       = []
        self._recording          = False
        self._record_buffer      = []
        self._current_holistic_feats: np.ndarray | None = None
        self._current_landmarks  = None
        self._current_hand_lms_list: list = []
        self._current_pose_lms   = None
        self._current_face_lms   = None
        self._latest_frame: np.ndarray | None = None
        self._auto_capturing     = False

        # ── Camera worker ─────────────────────────────────────────────────
        self._worker: CameraWorker | None = None

        # ── Auto-capture timer ────────────────────────────────────────────
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(AUTO_CAPTURE_INTERVAL)
        self._auto_timer.timeout.connect(self._auto_capture_step)

        self._overlay_widget = None
        self._build_ui()

        # ── Contact mode keyboard shortcut (window-scoped) ────────────────
        _sc = QShortcut(QKeySequence("C"), self)
        _sc.activated.connect(self._toggle_contact_mode)

    # ── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Nav bar
        nav = QWidget()
        nav.setFixedHeight(48)
        nav.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-bottom: 1px solid {HAIR};")
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(20, 0, 20, 0)
        nav_lay.setSpacing(12)

        back_btn = PillButton("← HOME", color=BG_ELEVATED,
                              hover_color=BG_FLOAT, text_color=TEXT_SEC)
        back_btn.setFixedWidth(100)
        back_btn.clicked.connect(self._on_back)
        nav_lay.addWidget(back_btn)

        title_lbl = QLabel("Training Mode")
        title_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {TEXT_TITLE};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;")
        nav_lay.addWidget(title_lbl, 1)

        self._hand_status = QLabel("")
        self._hand_status.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        nav_lay.addWidget(self._hand_status, 0, Qt.AlignmentFlag.AlignRight)

        root.addWidget(nav)

        # Body
        body = QWidget()
        body.setStyleSheet(f"background-color: {BG_DEEP};")
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(20, 14, 20, 14)
        body_lay.setSpacing(16)
        root.addWidget(body, 1)

        # ── Left: camera + thumbnails ─────────────────────────────────────
        left_lay = QVBoxLayout()
        left_lay.setSpacing(8)

        # Camera panel (flat, no shadow)
        cam_panel = QWidget()
        cam_panel.setStyleSheet(
            f"background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR};")
        cam_lay = QVBoxLayout(cam_panel)
        cam_lay.setContentsMargins(0, 0, 0, 0)
        self._cam_widget = CameraFrame(_CAM_W, _CAM_H, radius=0)
        cam_lay.addWidget(self._cam_widget)
        self._cam_card = cam_panel
        left_lay.addWidget(cam_panel)

        # Thumbnail strip
        thumb_label = QLabel("CAPTURED FRAMES")
        thumb_label.setStyleSheet(
            f"font-size: 9px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 2px;"
            " background: transparent; border: none;")
        left_lay.addWidget(thumb_label)

        thumb_scroll = QScrollArea()
        thumb_scroll.setFixedHeight(THUMB_SIZE + 16)
        thumb_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        thumb_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        thumb_scroll.setStyleSheet(
            f"background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR};")
        self._thumb_inner = QWidget()
        self._thumb_inner.setStyleSheet(f"background-color: {BG_SURFACE};")
        self._thumb_lay = QHBoxLayout(self._thumb_inner)
        self._thumb_lay.setContentsMargins(6, 4, 6, 4)
        self._thumb_lay.setSpacing(4)
        self._thumb_lay.addStretch(1)
        thumb_scroll.setWidget(self._thumb_inner)
        thumb_scroll.setWidgetResizable(True)
        left_lay.addWidget(thumb_scroll)

        body_lay.addLayout(left_lay, 3)

        # ── Right: controls card ──────────────────────────────────────────
        right_card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        right_card.setFixedWidth(256)
        right_lay = QVBoxLayout(right_card)
        right_lay.setContentsMargins(18, 18, 18, 18)
        right_lay.setSpacing(10)

        # Mode toggle
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(SectionLabel("Sign Type"))
        mode_row.addStretch(1)
        from PyQt6.QtWidgets import QRadioButton
        self._static_radio  = QRadioButton("Static")
        self._dynamic_radio = QRadioButton("Dynamic")
        self._static_radio.setChecked(True)
        self._static_radio.toggled.connect(
            lambda chk: chk and self._switch_mode("static"))
        self._dynamic_radio.toggled.connect(
            lambda chk: chk and self._switch_mode("dynamic"))
        mode_row.addWidget(self._static_radio)
        mode_row.addWidget(self._dynamic_radio)
        right_lay.addLayout(mode_row)

        right_lay.addWidget(HSep())

        # Contact mode indicator
        self._contact_mode_lbl = QLabel("Mode: Normal  [C to toggle]")
        self._contact_mode_lbl.setStyleSheet(
            f"font-size: 9px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        right_lay.addWidget(self._contact_mode_lbl)

        right_lay.addWidget(HSep())

        # Sign name
        right_lay.addWidget(SectionLabel("Sign Name"))
        from PyQt6.QtWidgets import QLineEdit
        self._name_entry = QLineEdit()
        self._name_entry.setPlaceholderText("e.g. hello")
        right_lay.addWidget(self._name_entry)

        right_lay.addSpacing(4)

        # Capture counter
        count_row = QHBoxLayout()
        count_row.addWidget(SectionLabel("Captured"))
        count_row.addStretch(1)
        self._count_lbl = QLabel(f"0 / {MAX_IMAGES}")
        self._count_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {TEXT_PRIMARY};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        count_row.addWidget(self._count_lbl)
        right_lay.addLayout(count_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setRange(0, MAX_IMAGES)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(_PROG_SS)
        right_lay.addWidget(self._progress_bar)

        self._start_btn = PillButton("▶  START CAPTURE", color=ACCENT)
        self._start_btn.clicked.connect(self._toggle_auto_capture)
        right_lay.addWidget(self._start_btn)

        self._auto_status = QLabel(
            f"Press Start — captures {AUTO_CAPTURE_TARGET} frames")
        self._auto_status.setWordWrap(True)
        self._auto_status.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        right_lay.addWidget(self._auto_status)

        right_lay.addWidget(HSep())

        self._save_btn = PillButton("SAVE & TRAIN ✓", color=SUCCESS,
                                    text_color="#0d1a11")
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setEnabled(False)
        right_lay.addWidget(self._save_btn)

        cancel_btn = DangerButton("CANCEL")
        cancel_btn.clicked.connect(self._on_back)
        right_lay.addWidget(cancel_btn)

        # ── Dynamic panel (hidden initially) ─────────────────────────────
        self._dyn_widget = QWidget()
        self._dyn_widget.setStyleSheet("background: transparent;")
        dyn_lay = QVBoxLayout(self._dyn_widget)
        dyn_lay.setContentsMargins(0, 0, 0, 0)
        dyn_lay.setSpacing(8)

        seq_row = QHBoxLayout()
        seq_row.addWidget(SectionLabel("Sequences"))
        seq_row.addStretch(1)
        self._seq_count_lbl = QLabel(f"0 / {MIN_SEQS} min")
        self._seq_count_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {TEXT_PRIMARY};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        seq_row.addWidget(self._seq_count_lbl)
        dyn_lay.addLayout(seq_row)

        self._rec_bar = QProgressBar()
        self._rec_bar.setFixedHeight(4)
        self._rec_bar.setRange(0, SEQ_LENGTH)
        self._rec_bar.setValue(0)
        self._rec_bar.setTextVisible(False)
        self._rec_bar.setStyleSheet(_PROG_SS)
        dyn_lay.addWidget(self._rec_bar)

        self._rec_status_lbl = QLabel(
            f"Hold gesture and click Record ({SEQ_LENGTH} frames each)")
        self._rec_status_lbl.setWordWrap(True)
        self._rec_status_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        dyn_lay.addWidget(self._rec_status_lbl)

        self._record_btn = PillButton("⏺  RECORD SEQUENCE", color=ACCENT)
        self._record_btn.clicked.connect(self._on_record)
        dyn_lay.addWidget(self._record_btn)

        dyn_lay.addWidget(HSep())

        self._dyn_save_btn = PillButton("SAVE & TRAIN LSTM ✓", color=SUCCESS,
                                        text_color="#0d1a11")
        self._dyn_save_btn.clicked.connect(self._on_save_dynamic)
        self._dyn_save_btn.setEnabled(False)
        dyn_lay.addWidget(self._dyn_save_btn)

        self._dyn_hint_lbl = QLabel(
            f"Record at least {MIN_SEQS} sequences to unlock Save")
        self._dyn_hint_lbl.setWordWrap(True)
        self._dyn_hint_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        dyn_lay.addWidget(self._dyn_hint_lbl)

        self._dyn_widget.setVisible(False)
        right_lay.addWidget(self._dyn_widget)
        right_lay.addStretch(1)

        body_lay.addWidget(right_card, 0)

    # ── Training overlay ─────────────────────────────────────────────────────
    def show_training_overlay(self, text: str):
        if self._overlay_widget:
            return
        overlay = QWidget(self)
        overlay.setStyleSheet(f"background-color: rgba(10,10,9,210);")
        overlay.resize(self.size())
        lay = QVBoxLayout(overlay)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"font-size: 20px; font-weight: 400; color: {SUCCESS};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;")
        sub = QLabel("Please wait…")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(
            f"font-size: 11px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        lay.addWidget(lbl)
        lay.addWidget(sub)
        overlay.show()
        self._overlay_widget = overlay

    def hide_training_overlay(self):
        if self._overlay_widget:
            self._overlay_widget.deleteLater()
            self._overlay_widget = None

    def resizeEvent(self, e):
        if self._overlay_widget:
            self._overlay_widget.resize(self.size())
        super().resizeEvent(e)

    # ── Public API ───────────────────────────────────────────────────────────
    def activate(self):
        print("[TrainingScreen] Activated")
        self._start_camera()

    def deactivate(self):
        print("[TrainingScreen] Deactivated")
        self._stop_auto_capture()
        self._stop_camera()

    def reset(self):
        self._stop_auto_capture()
        self.captured_features.clear()
        self.captured_seqs.clear()
        self._record_buffer.clear()
        self._recording = False
        self._current_landmarks = None
        self._current_hand_lms_list = []
        self._current_pose_lms = None
        self._current_face_lms = None
        self._current_holistic_feats = None
        self._latest_frame = None
        self._name_entry.clear()
        while self._thumb_lay.count() > 1:
            item = self._thumb_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._update_count()
        self._update_seq_count()
        self._save_btn.setEnabled(False)
        self._dyn_save_btn.setEnabled(False)
        self._auto_status.setText(
            f"Press Start — captures {AUTO_CAPTURE_TARGET} frames")
        self._auto_status.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")

    def has_unsaved_data(self) -> bool:
        return bool(self.captured_features) or bool(self.captured_seqs)

    def cleanup(self):
        self._stop_camera()
        if self._owns_resources:
            try:
                self._detector.close()
            except Exception:
                pass
        if self._holistic is not None and self._owns_holistic:
            try:
                self._holistic.close()
            except Exception:
                pass
            self._holistic = None

    # ── Camera (worker-based) ─────────────────────────────────────────────────
    def _start_camera(self):
        ok = self._camera.start()
        if not ok:
            self._set_hand_status("Camera not found", DANGER)
            print("[TrainingScreen] Camera not found")
            return

        self._set_hand_status("Camera starting…", TEXT_SEC)

        self._worker = CameraWorker(
            camera=self._camera,
            detector=self._detector,
            holistic=self._holistic,
            pose_detector=self._pose_detector,
            parent=self,
        )
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.start_capture()
        print("[TrainingScreen] Camera worker started")

    def _stop_camera(self):
        if self._worker is not None:
            self._worker.stop_capture()
            self._worker = None
        self._camera.stop()
        self._current_landmarks = None
        self._latest_frame = None
        print("[TrainingScreen] Camera stopped")

    # ── Frame handler (main thread — called via queued signal) ───────────────
    def _on_frame_ready(self, annotated_frame, hand_lms_list,
                         pose_lms, face_lms, holistic_feats):
        """Receives detection results from CameraWorker; updates UI only."""
        self._current_landmarks     = hand_lms_list[0] if hand_lms_list else None
        self._current_hand_lms_list = hand_lms_list
        self._current_pose_lms      = pose_lms
        self._current_face_lms      = face_lms
        self._current_holistic_feats = holistic_feats
        self._latest_frame = annotated_frame

        # Draw contact mode reminder overlay before displaying
        display_frame = annotated_frame.copy()
        if self._contact_mode:
            self._draw_contact_mode_overlay(display_frame)
        self._cam_widget.set_frame(cv2_to_qpixmap(display_frame))

        if hand_lms_list:
            n   = len(hand_lms_list)
            msg = "Hand detected" if n == 1 else f"{n} hands detected"
            self._set_hand_status(msg, SUCCESS)
            if self._contact_mode:
                self._cam_card.setStyleSheet(
                    f"background-color: {BG_SURFACE};"
                    f" border: 1px solid {lerp(BG_BORDER, WARNING, 0.6)};")
            else:
                self._cam_card.setStyleSheet(
                    f"background-color: {BG_SURFACE};"
                    f" border: 1px solid {lerp(BG_BORDER, SUCCESS, 0.4)};")
        else:
            self._set_hand_status("No hand detected", TEXT_SEC)
            self._cam_card.setStyleSheet(
                f"background-color: {BG_SURFACE};"
                f" border: 1px solid {HAIR};")

        if self._mode == "dynamic" and self._recording:
            if self._current_holistic_feats is not None:
                self._record_buffer.append(self._current_holistic_feats.copy())
                progress = len(self._record_buffer)
                self._rec_bar.setValue(progress)
                remaining = SEQ_LENGTH - progress
                self._rec_status_lbl.setText(
                    f"Recording… {remaining} frames remaining")
                self._rec_status_lbl.setStyleSheet(
                    f"font-size: 10px; color: {ACCENT};"
                    f" font-family: 'Courier New', monospace;"
                    " background: transparent; border: none;")
                if progress >= SEQ_LENGTH:
                    seq = np.array(
                        self._record_buffer[:SEQ_LENGTH], dtype=np.float32)
                    self.captured_seqs.append(seq)
                    self._record_buffer.clear()
                    self._recording = False
                    self._record_btn.setEnabled(True)
                    self._update_seq_count()
                    self._cam_card.setStyleSheet(
                        f"background-color: {BG_SURFACE};"
                        f" border: 1px solid {lerp(BG_BORDER, SUCCESS, 0.6)};")
                    QTimer.singleShot(
                        300,
                        lambda: self._cam_card.setStyleSheet(
                            f"background-color: {BG_SURFACE};"
                            f" border: 1px solid {HAIR};"))

    # ── Auto-capture ──────────────────────────────────────────────────────────
    def _toggle_auto_capture(self):
        if self._auto_capturing:
            self._stop_auto_capture()
        else:
            self._start_auto_capture()

    def _start_auto_capture(self):
        if self._current_landmarks is None and self._holistic is None:
            QMessageBox.information(
                self, "No Hand Detected",
                "Hold your hand in front of the camera\n"
                "and wait for the green indicator.")
            return
        self._auto_capturing = True
        self._start_btn.setText("⏹  STOP CAPTURE")
        self._auto_timer.start()

    def _stop_auto_capture(self):
        self._auto_capturing = False
        self._auto_timer.stop()
        self._start_btn.setText("▶  START CAPTURE")

    def _auto_capture_step(self):
        if not self._auto_capturing:
            self._stop_auto_capture()
            return

        count = len(self.captured_features)
        if count >= AUTO_CAPTURE_TARGET:
            self._stop_auto_capture()
            self._auto_status.setText(
                f"Done  ·  {count} frames captured — ready to Save & Train")
            self._auto_status.setStyleSheet(
                f"font-size: 10px; color: {SUCCESS};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;")
            return

        lms = self._current_landmarks
        if lms is not None:
            features = self._spatial_extractor.extract_from_holistic(
                self._current_hand_lms_list,
                self._current_pose_lms,
                self._current_face_lms,
            )
            if not self._shape_printed:
                print(f"[TrainingScreen] New spatial feature vector shape: "
                      f"{features.shape}  (was 63 raw landmarks)")
                from core.feature_extractor import (
                    SPATIAL_FEATURE_LENGTH, _B_PER_HAND, _C_PER_HAND,
                    _D_PER_ARM, _G_PER_HAND,
                )
                print(
                    f"  Group B (hand position × 2):  {_B_PER_HAND * 2:>3}\n"
                    f"  Group C (joint angles × 2):   {_C_PER_HAND * 2:>3}\n"
                    f"  Group D (arm position × 2):   {_D_PER_ARM  * 2:>3}\n"
                    f"  Group E (dominant hand):        1\n"
                    f"  Group F (hand-to-hand):         3\n"
                    f"  Group G (contact × 2 + flag): {_G_PER_HAND * 2 + 1:>3}\n"
                    f"  ─────────────────────────────────\n"
                    f"  Total:                        {SPATIAL_FEATURE_LENGTH:>3}\n"
                    f"\n"
                    f"  ⚠  You must now RETRAIN the model on new data.\n"
                    f"     Record each sign ≥30 times normally AND ≥30 times\n"
                    f"     with hand touching face where relevant\n"
                    f"     (THANK_YOU, PLEASE, SORRY, etc.).\n"
                    f"     Press 'C' in Training Mode to toggle CONTACT reminder."
                )
                self._shape_printed = True
            self.captured_features.append(features)
            if self._latest_frame is not None:
                self._add_thumbnail(self._latest_frame)
            self._update_count()
            self._auto_status.setText(
                f"Capturing…  {len(self.captured_features)} / {AUTO_CAPTURE_TARGET}")
            self._auto_status.setStyleSheet(
                f"font-size: 10px; color: {ACCENT};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;")
        else:
            self._auto_status.setText(
                f"Waiting for hand…  {count} / {AUTO_CAPTURE_TARGET}")

    # ── Thumbnails ────────────────────────────────────────────────────────────
    def _add_thumbnail(self, bgr_frame: np.ndarray):
        from PyQt6.QtGui import QPixmap, QImage
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(bytes(rgb.data), w, h, ch * w,
                      QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            THUMB_SIZE, THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation)
        lbl = QLabel(self._thumb_inner)
        lbl.setPixmap(pix)
        lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        lbl.setStyleSheet(
            f"border: 1px solid rgba(74,124,89,0.5); background: transparent;")
        self._thumb_lay.insertWidget(self._thumb_lay.count() - 1, lbl)

    # ── Widget helpers ────────────────────────────────────────────────────────
    def _set_hand_status(self, text: str, color: str):
        self._hand_status.setText(text)
        self._hand_status.setStyleSheet(
            f"font-size: 10px; color: {color};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")

    def _update_count(self):
        n = len(self.captured_features)
        self._count_lbl.setText(f"{n} / {MAX_IMAGES}")
        self._progress_bar.setValue(n)
        if n >= MIN_IMAGES:
            self._save_btn.setEnabled(True)
            self._progress_bar.setStyleSheet(_PROG_SS_OK)
        else:
            self._progress_bar.setStyleSheet(_PROG_SS)

    def _update_seq_count(self):
        n = len(self.captured_seqs)
        self._seq_count_lbl.setText(f"{n} / {MIN_SEQS} min")
        if n >= MIN_SEQS:
            self._dyn_save_btn.setEnabled(True)
            self._dyn_hint_lbl.setText(
                f"{n} sequences — ready to train LSTM!")
            self._dyn_hint_lbl.setStyleSheet(
                f"font-size: 10px; color: {SUCCESS};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;")
        else:
            self._dyn_hint_lbl.setText(
                f"Record {MIN_SEQS - n} more to unlock Save")
            self._dyn_hint_lbl.setStyleSheet(
                f"font-size: 10px; color: {TEXT_HINT};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;")

    # ── Mode switching ────────────────────────────────────────────────────────
    def _switch_mode(self, mode: str):
        self._mode = mode
        if mode == "dynamic":
            self._stop_auto_capture()
            self._init_holistic()
            if self._worker is not None:
                self._worker.stop_capture()
                self._worker = CameraWorker(
                    camera=self._camera,
                    detector=self._detector,
                    holistic=self._holistic,
                    pose_detector=self._pose_detector,
                    parent=self,
                )
                self._worker.frame_ready.connect(self._on_frame_ready)
                self._worker.start_capture()
            self._start_btn.setVisible(False)
            self._auto_status.setVisible(False)
            self._save_btn.setVisible(False)
            self._dyn_widget.setVisible(True)
            self._update_seq_count()
        else:
            self._dyn_widget.setVisible(False)
            self._start_btn.setVisible(True)
            self._auto_status.setVisible(True)
            self._save_btn.setVisible(True)

    def _init_holistic(self):
        if self._holistic is not None:
            return
        try:
            from core.holistic_detector import HolisticDetector
            self._holistic = HolisticDetector()
            self._owns_holistic = True
            print("[TrainingScreen] Holistic detector initialised")
        except Exception as exc:
            QMessageBox.critical(
                self, "Holistic Model Error",
                f"Could not load holistic detector:\n{exc}\n\n"
                "Falling back to Static mode.")
            self._static_radio.setChecked(True)
            self._mode = "static"

    # ── Record ────────────────────────────────────────────────────────────────
    def _on_record(self):
        if self._holistic is None:
            QMessageBox.information(
                self, "Not Ready", "Holistic detector is not initialised.")
            return
        if len(self.captured_seqs) >= MAX_SEQS:
            QMessageBox.information(
                self, "Maximum Reached",
                f"You have recorded {MAX_SEQS} sequences.\n"
                "Click Save & Train LSTM to continue.")
            return
        self._recording = True
        self._record_buffer.clear()
        self._record_btn.setEnabled(False)
        self._rec_bar.setValue(0)
        self._rec_status_lbl.setText(
            f"Recording… {SEQ_LENGTH} frames remaining")
        self._rec_status_lbl.setStyleSheet(
            f"font-size: 10px; color: {ACCENT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;")
        print("[TrainingScreen] Recording started")

    # ── Save ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _sanitize_name(raw: str) -> str:
        name = raw.strip().replace(" ", "_").lower()
        name = re.sub(r"[^a-z0-9_]", "", name)
        return name[:50]

    def _on_save(self):
        sign_name = self._sanitize_name(self._name_entry.text())
        if not sign_name:
            QMessageBox.warning(
                self, "Invalid Name",
                "Please enter a sign name using letters, numbers, or spaces.\n\n"
                "Examples: 'Hello', 'Thank You', 'Stop'")
            return
        if len(self.captured_features) < MIN_IMAGES:
            remaining = MIN_IMAGES - len(self.captured_features)
            QMessageBox.warning(
                self, "Not Enough Data",
                f"You need at least {MIN_IMAGES} samples to train.\n\n"
                f"Captured: {len(self.captured_features)}\n"
                f"Remaining: {remaining} more needed")
            return
        self._stop_camera()
        if self.on_save:
            self.on_save(sign_name, self.captured_features)

    def _on_save_dynamic(self):
        sign_name = self._sanitize_name(self._name_entry.text())
        if not sign_name:
            QMessageBox.warning(
                self, "Invalid Name",
                "Please enter a sign name (letters, numbers, spaces).")
            return
        if len(self.captured_seqs) < MIN_SEQS:
            QMessageBox.warning(
                self, "Not Enough Sequences",
                f"Record at least {MIN_SEQS} sequences.\n"
                f"Current: {len(self.captured_seqs)}")
            return
        self._stop_camera()
        if self.on_save_sequences:
            self.on_save_sequences(sign_name, list(self.captured_seqs))

    def _on_back(self):
        self._stop_auto_capture()
        self._stop_camera()
        self.reset()
        if self.on_back:
            self.on_back()

    # ── Contact mode ─────────────────────────────────────────────────────────
    def _toggle_contact_mode(self):
        self._contact_mode = not self._contact_mode
        if self._contact_mode:
            self._contact_mode_lbl.setText(
                "Mode: CONTACT — hand touching face  [C]")
            self._contact_mode_lbl.setStyleSheet(
                f"font-size: 9px; color: {WARNING};"
                f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
                " background: transparent; border: none;")
        else:
            self._contact_mode_lbl.setText("Mode: Normal  [C to toggle]")
            self._contact_mode_lbl.setStyleSheet(
                f"font-size: 9px; color: {TEXT_HINT};"
                f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
                " background: transparent; border: none;")

    def _draw_contact_mode_overlay(self, frame: np.ndarray) -> None:
        """Draw amber 'CONTACT SIGN' reminder overlay on the camera frame."""
        h, w = frame.shape[:2]
        # Amber border  (WARNING = #b8924a → BGR: 74, 146, 184)
        _AMBER_BGR = (74, 146, 184)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _AMBER_BGR, 3)
        cv2.putText(
            frame, "CONTACT SIGN",
            (10, h - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, _AMBER_BGR, 2, cv2.LINE_AA,
        )
