"""
Train dialog for SignBridge — multi-step recording flow.

Replaces the legacy ``TrainingScreen``. Supports:
  1. Label entry         — sanitize → single word vs multi-word phrase
  2. Walk-through        — multi-word phrases step through one word at a time
  3. Countdown           — 3-2-1-GO full-screen overlay (cancelable with Escape)
  4. Capture             — auto-capture 30 frames at 300ms intervals
  5. Save                — emits save_requested(sign_name, captured_features)
  6. Phrase persistence  — multi-word phrases get appended to phrases.json

The capture and frame handling logic is intentionally copied from the legacy
TrainingScreen rather than refactored out, to honour the wrap-and-extend
constraint and minimise risk of regressing the existing screen.
"""
from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from config import SAMPLES_PER_CLASS
from core import phrase_store
from core.camera_worker import CameraWorker
from core.feature_extractor import SpatialFeatureExtractor
from gui.countdown_overlay import CountdownOverlay
from gui.design import (
    ACCENT, BG_BASE, BG_BORDER, BG_DEEP, BG_FLOAT, BG_SURFACE, HAIR,
    SUCCESS, DANGER, TEXT_HINT, TEXT_PRIMARY, TEXT_SEC, TEXT_TITLE,
    CameraFrame, GlassCard, HSep, OutlineButton, PillButton, DangerButton,
    SectionLabel, cv2_to_qpixmap, lerp,
)


# ─── Capture constants ────────────────────────────────────────────────────────
MIN_IMAGES            = 10
MAX_IMAGES            = SAMPLES_PER_CLASS
THUMB_SIZE            = 60
AUTO_CAPTURE_INTERVAL = 300       # ms
AUTO_CAPTURE_TARGET   = 30

_CAM_W = 580
_CAM_H = 435


def _sanitize_name(raw: str) -> str:
    """Lowercase + underscores + alnum-only, max 50 chars."""
    name = raw.strip().replace(" ", "_").lower()
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name[:50]


class TrainDialog(QWidget):
    """Multi-step training page added to the MainWindow stack.

    Signals:
        save_requested(sign_name, captured_features):
            Emitted when the user finishes capturing the current word/sign
            and the data is ready to be persisted by RFTrainWorker.
        back_requested():
            Emitted when the user cancels out to the prediction screen.
    """

    save_requested = pyqtSignal(str, list)
    back_requested = pyqtSignal()

    def __init__(self, camera, holistic, parent=None):
        super().__init__(parent)
        self._camera   = camera
        self._holistic = holistic

        self._spatial_extractor = SpatialFeatureExtractor()

        # ── Capture state ───────────────────────────────────────────────────
        self._captured_features: list[np.ndarray] = []
        self._current_hand_lms_list: list = []
        self._current_pose_lms = None
        self._current_face_lms = None
        self._latest_frame: Optional[np.ndarray] = None
        self._auto_capturing = False

        # ── Walk-through state ──────────────────────────────────────────────
        self._words: list[str] = []
        self._word_idx: int = 0
        self._original_input: str = ""

        # ── Camera worker (created on demand) ───────────────────────────────
        self._worker: Optional[CameraWorker] = None

        # ── Auto-capture timer ──────────────────────────────────────────────
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(AUTO_CAPTURE_INTERVAL)
        self._auto_timer.timeout.connect(self._auto_capture_step)

        self._build_ui()

    # ─── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Nav bar
        nav = QWidget()
        nav.setFixedHeight(48)
        nav.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-bottom: 1px solid {HAIR};"
        )
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(20, 0, 20, 0)
        nav_lay.setSpacing(12)

        back_btn = OutlineButton("← BACK")
        back_btn.setFixedWidth(100)
        back_btn.clicked.connect(self._on_back)
        nav_lay.addWidget(back_btn)

        title_lbl = QLabel("Train")
        title_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {TEXT_TITLE};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;"
        )
        nav_lay.addWidget(title_lbl, 1)

        self._step_lbl = QLabel("")
        self._step_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;"
        )
        nav_lay.addWidget(self._step_lbl, 0, Qt.AlignmentFlag.AlignRight)
        root.addWidget(nav)

        # Stacked body: 0=label entry, 1=capture
        self._body = QStackedWidget()
        self._body.setStyleSheet(f"background-color: {BG_DEEP};")
        root.addWidget(self._body, 1)

        self._build_label_step()
        self._build_capture_step()

        # Countdown overlay (covers the whole TrainDialog widget)
        self._countdown = CountdownOverlay(self)
        self._countdown.finished.connect(self._on_countdown_finished)
        self._countdown.canceled.connect(self._on_countdown_canceled)

    def _build_label_step(self):
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(40, 60, 40, 40)
        page_lay.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        page_lay.setSpacing(0)

        card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        card.setFixedWidth(560)
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(40, 36, 40, 36)
        card_lay.setSpacing(16)

        heading = QLabel("Record a new sign")
        heading.setStyleSheet(
            f"font-size: 22px; color: {TEXT_TITLE};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;"
        )
        card_lay.addWidget(heading)

        sub = QLabel(
            "Type a single word like 'hello' or a phrase like "
            "'nice to meet you'. Phrases will walk you through one word "
            "at a time."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"font-size: 11px; color: {TEXT_SEC};"
            f" font-family: 'Courier New', monospace; line-height: 1.6;"
            " background: transparent; border: none;"
        )
        card_lay.addWidget(sub)

        card_lay.addSpacing(8)
        card_lay.addWidget(SectionLabel("Sign or phrase"))

        self._name_entry = QLineEdit()
        self._name_entry.setPlaceholderText("e.g. hello   or   nice to meet you")
        self._name_entry.returnPressed.connect(self._on_begin)
        card_lay.addWidget(self._name_entry)

        card_lay.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._begin_btn = PillButton("BEGIN", color=ACCENT)
        self._begin_btn.clicked.connect(self._on_begin)
        btn_row.addWidget(self._begin_btn)
        card_lay.addLayout(btn_row)

        page_lay.addWidget(card)
        self._body.addWidget(page)

    def _build_capture_step(self):
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(20, 14, 20, 14)
        page_lay.setSpacing(0)

        # Header strip — current word indicator
        self._word_header = QLabel("")
        self._word_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._word_header.setStyleSheet(
            f"font-size: 18px; color: {TEXT_TITLE};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            f" background: transparent; border: none; padding: 6px;"
        )
        page_lay.addWidget(self._word_header)

        body_lay = QHBoxLayout()
        body_lay.setSpacing(16)
        page_lay.addLayout(body_lay, 1)

        # ── Left: camera + thumbnails ─────────────────────────────────────
        left_lay = QVBoxLayout()
        left_lay.setSpacing(8)

        cam_panel = QWidget()
        cam_panel.setStyleSheet(
            f"background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR};"
        )
        cam_lay = QVBoxLayout(cam_panel)
        cam_lay.setContentsMargins(0, 0, 0, 0)
        self._cam_widget = CameraFrame(_CAM_W, _CAM_H, radius=0)
        cam_lay.addWidget(self._cam_widget)
        self._cam_card = cam_panel
        left_lay.addWidget(cam_panel)

        thumb_label = QLabel("CAPTURED FRAMES")
        thumb_label.setStyleSheet(
            f"font-size: 9px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 2px;"
            " background: transparent; border: none;"
        )
        left_lay.addWidget(thumb_label)

        thumb_scroll = QScrollArea()
        thumb_scroll.setFixedHeight(THUMB_SIZE + 16)
        thumb_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        thumb_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        thumb_scroll.setStyleSheet(
            f"background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR};"
        )
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

        # ── Right: status card ─────────────────────────────────────────────
        right_card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        right_card.setFixedWidth(256)
        right_lay = QVBoxLayout(right_card)
        right_lay.setContentsMargins(18, 18, 18, 18)
        right_lay.setSpacing(10)

        right_lay.addWidget(SectionLabel("Status"))

        self._hand_status = QLabel("Waiting for camera…")
        self._hand_status.setWordWrap(True)
        self._hand_status.setStyleSheet(
            f"font-size: 11px; color: {TEXT_SEC};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;"
        )
        right_lay.addWidget(self._hand_status)

        right_lay.addWidget(HSep())

        right_lay.addWidget(SectionLabel("Captured"))
        count_row = QHBoxLayout()
        count_row.addStretch(1)
        self._count_lbl = QLabel(f"0 / {AUTO_CAPTURE_TARGET}")
        self._count_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {TEXT_PRIMARY};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;"
        )
        count_row.addWidget(self._count_lbl)
        right_lay.addLayout(count_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setRange(0, AUTO_CAPTURE_TARGET)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {HAIR};"
            f" height: 4px; border-radius: 0; }}"
            f"QProgressBar::chunk {{ background: {ACCENT}; border-radius: 0; }}"
        )
        right_lay.addWidget(self._progress_bar)

        self._auto_status = QLabel("Get ready…")
        self._auto_status.setWordWrap(True)
        self._auto_status.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;"
        )
        right_lay.addWidget(self._auto_status)

        right_lay.addWidget(HSep())

        # Confirmation strip — populated after save
        self._confirm_lbl = QLabel("")
        self._confirm_lbl.setWordWrap(True)
        self._confirm_lbl.setStyleSheet(
            f"font-size: 11px; color: {SUCCESS};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;"
        )
        right_lay.addWidget(self._confirm_lbl)

        cancel_btn = DangerButton("CANCEL")
        cancel_btn.clicked.connect(self._on_back)
        right_lay.addWidget(cancel_btn)

        right_lay.addStretch(1)

        body_lay.addWidget(right_card, 0)

        self._body.addWidget(page)

    # ─── Public API ───────────────────────────────────────────────────────────
    def activate(self, prefill: str | None = None) -> None:
        """Switch to the label-entry step. Optionally pre-fill the entry.

        If ``prefill`` is non-empty, the label-entry step is skipped and
        the dialog jumps straight into recording for that single sign
        (used by the View "Add samples" flow).
        """
        self._reset_capture_state()
        self._confirm_lbl.setText("")
        if prefill:
            sanitized = _sanitize_name(prefill)
            self._name_entry.setText(sanitized)
            self._words = [sanitized]
            self._word_idx = 0
            self._original_input = sanitized
            self._show_capture_step()
            self._start_word()
        else:
            self._words = []
            self._word_idx = 0
            self._original_input = ""
            self._name_entry.clear()
            self._step_lbl.setText("")
            self._body.setCurrentIndex(0)
            self._name_entry.setFocus()

    def deactivate(self) -> None:
        """Stop everything when leaving the dialog."""
        self._stop_auto_capture()
        self._stop_camera()
        self._countdown.cancel()

    def has_unsaved_data(self) -> bool:
        return bool(self._captured_features)

    def cleanup(self) -> None:
        self._stop_auto_capture()
        self._stop_camera()

    def on_save_complete(self, result: dict) -> None:
        """Called by MainWindow when the RFTrainWorker finishes saving.

        Result dict shape (from RFTrainWorker.finished):
            {sign_name, num_saved, accuracy, num_samples}
        """
        sign  = result.get("sign_name", "?")
        saved = int(result.get("num_saved", 0))
        display = sign.replace("_", " ").upper()
        self._confirm_lbl.setText(f"Saved: {saved} samples for {display}")

        # Multi-word phrase: advance to next word, otherwise we're done
        if self._word_idx + 1 < len(self._words):
            self._word_idx += 1
            QTimer.singleShot(700, self._start_word)
        else:
            # End of phrase — persist phrases.json if multi-word
            if len(self._words) > 1:
                try:
                    phrase_store.add_phrase(
                        sequence=self._words,
                        output=" ".join(self._words),
                    )
                except Exception as exc:
                    print(f"[TrainDialog] phrase persist failed: {exc}")
            QTimer.singleShot(900, self._finish_flow)

    def on_save_error(self, msg: str) -> None:
        """Called by MainWindow if the RFTrainWorker raised an error."""
        self._confirm_lbl.setStyleSheet(
            f"font-size: 11px; color: {DANGER};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;"
        )
        self._confirm_lbl.setText(f"Error: {msg}")
        QTimer.singleShot(2000, self._finish_flow)

    # ─── Step transitions ─────────────────────────────────────────────────────
    def _on_begin(self) -> None:
        raw = self._name_entry.text().strip()
        if not raw:
            QMessageBox.warning(
                self, "Invalid Name",
                "Please enter at least one word using letters or numbers.",
            )
            return

        self._original_input = raw
        # Split first, then sanitize each token (preserves multi-word intent)
        tokens = [_sanitize_name(t) for t in raw.split() if _sanitize_name(t)]
        if not tokens:
            QMessageBox.warning(
                self, "Invalid Name",
                "After sanitizing, no usable words remain.\n"
                "Use letters or numbers only.",
            )
            return

        self._words = tokens
        self._word_idx = 0
        self._show_capture_step()
        self._start_word()

    def _show_capture_step(self) -> None:
        self._body.setCurrentIndex(1)

    def _start_word(self) -> None:
        """Begin the countdown → capture cycle for ``self._words[self._word_idx]``."""
        self._reset_capture_state()
        self._confirm_lbl.setText("")
        self._confirm_lbl.setStyleSheet(
            f"font-size: 11px; color: {SUCCESS};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;"
        )

        word = self._words[self._word_idx].replace("_", " ").upper()
        if len(self._words) > 1:
            self._word_header.setText(
                f"Word {self._word_idx + 1} of {len(self._words)}  —  {word}"
            )
            self._step_lbl.setText(
                f"Step {self._word_idx + 1}/{len(self._words)}"
            )
        else:
            self._word_header.setText(word)
            self._step_lbl.setText("")

        self._auto_status.setText("Get ready…")
        self._countdown.setGeometry(self.rect())
        self._countdown.start()

    def _on_countdown_finished(self) -> None:
        """Countdown reached GO — start the camera + auto-capture."""
        self._start_camera()
        self._start_auto_capture()

    def _on_countdown_canceled(self) -> None:
        """User pressed Escape during countdown — return to label entry."""
        self._stop_camera()
        self._words = []
        self._word_idx = 0
        self._body.setCurrentIndex(0)
        self._step_lbl.setText("")
        self._name_entry.setFocus()

    def _finish_flow(self) -> None:
        """End-of-phrase: stop everything and return to caller."""
        self._stop_auto_capture()
        self._stop_camera()
        self.back_requested.emit()

    def _on_back(self) -> None:
        self._stop_auto_capture()
        self._stop_camera()
        self._countdown.cancel()
        self.back_requested.emit()

    # ─── Camera (worker-based) ────────────────────────────────────────────────
    def _start_camera(self) -> None:
        if self._worker is not None:
            return
        ok = self._camera.start()
        if not ok:
            self._set_hand_status("Camera not found", DANGER)
            return
        self._set_hand_status("Camera starting…", TEXT_SEC)
        self._worker = CameraWorker(
            camera=self._camera,
            detector=None,
            holistic=self._holistic,
            pose_detector=None,
            parent=self,
        )
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.start_capture()

    def _stop_camera(self) -> None:
        if self._worker is not None:
            try:
                self._worker.stop_capture()
            except Exception:
                pass
            self._worker = None
        try:
            self._camera.stop()
        except Exception:
            pass
        self._latest_frame = None

    # ─── Frame handler (main thread, queued signal) ───────────────────────────
    def _on_frame_ready(self, annotated_frame, hand_lms_list,
                         pose_lms, face_lms, _holistic_feats) -> None:
        self._current_hand_lms_list = hand_lms_list
        self._current_pose_lms      = pose_lms
        self._current_face_lms      = face_lms
        self._latest_frame          = annotated_frame

        display_frame = annotated_frame.copy()
        if hand_lms_list and pose_lms is not None:
            features = self._spatial_extractor.extract_from_holistic(
                hand_lms_list, pose_lms, face_lms)
            self._spatial_extractor.draw_debug(
                display_frame, features, pose_lms,
                face_lms=face_lms, hand_lms_list=hand_lms_list,
            )
        self._cam_widget.set_frame(cv2_to_qpixmap(display_frame))

        if hand_lms_list:
            n = len(hand_lms_list)
            msg = "Hand detected" if n == 1 else f"{n} hands detected"
            self._set_hand_status(msg, SUCCESS)
            self._cam_card.setStyleSheet(
                f"background-color: {BG_SURFACE};"
                f" border: 1px solid {lerp(BG_BORDER, SUCCESS, 0.4)};"
            )
        else:
            self._set_hand_status("No hand detected", TEXT_SEC)
            self._cam_card.setStyleSheet(
                f"background-color: {BG_SURFACE};"
                f" border: 1px solid {HAIR};"
            )

    # ─── Auto-capture ─────────────────────────────────────────────────────────
    def _start_auto_capture(self) -> None:
        self._auto_capturing = True
        self._auto_timer.start()

    def _stop_auto_capture(self) -> None:
        self._auto_capturing = False
        self._auto_timer.stop()

    def _auto_capture_step(self) -> None:
        if not self._auto_capturing:
            return

        count = len(self._captured_features)
        if count >= AUTO_CAPTURE_TARGET:
            self._stop_auto_capture()
            self._auto_status.setText(
                f"Done · {count} frames captured — saving…"
            )
            self._auto_status.setStyleSheet(
                f"font-size: 10px; color: {SUCCESS};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;"
            )
            self._stop_camera()
            self.save_requested.emit(
                self._words[self._word_idx],
                list(self._captured_features),
            )
            return

        if self._current_hand_lms_list:
            features = self._spatial_extractor.extract_from_holistic(
                self._current_hand_lms_list,
                self._current_pose_lms,
                self._current_face_lms,
            )
            self._captured_features.append(features)
            if self._latest_frame is not None:
                self._add_thumbnail(self._latest_frame)
            self._update_count()
            self._auto_status.setText(
                f"Capturing… {len(self._captured_features)} / {AUTO_CAPTURE_TARGET}"
            )
            self._auto_status.setStyleSheet(
                f"font-size: 10px; color: {ACCENT};"
                f" font-family: 'Courier New', monospace;"
                " background: transparent; border: none;"
            )
        else:
            self._auto_status.setText(
                f"Waiting for hand… {count} / {AUTO_CAPTURE_TARGET}"
            )

    # ─── Thumbnails ───────────────────────────────────────────────────────────
    def _add_thumbnail(self, bgr_frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(bytes(rgb.data), w, h, ch * w,
                      QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            THUMB_SIZE, THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        lbl = QLabel(self._thumb_inner)
        lbl.setPixmap(pix)
        lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        lbl.setStyleSheet(
            f"border: 1px solid rgba(74,124,89,0.5); background: transparent;"
        )
        self._thumb_lay.insertWidget(self._thumb_lay.count() - 1, lbl)

    # ─── State helpers ────────────────────────────────────────────────────────
    def _reset_capture_state(self) -> None:
        self._stop_auto_capture()
        self._captured_features.clear()
        self._current_hand_lms_list = []
        self._current_pose_lms = None
        self._current_face_lms = None
        self._latest_frame = None
        self._update_count()
        self._set_hand_status("Waiting for camera…", TEXT_SEC)
        # Clear thumbnails
        while self._thumb_lay.count() > 1:
            item = self._thumb_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumb_lay.addStretch(1)

    def _set_hand_status(self, text: str, color: str) -> None:
        self._hand_status.setText(text)
        self._hand_status.setStyleSheet(
            f"font-size: 11px; color: {color};"
            f" font-family: 'Courier New', monospace;"
            " background: transparent; border: none;"
        )

    def _update_count(self) -> None:
        n = len(self._captured_features)
        self._count_lbl.setText(f"{n} / {AUTO_CAPTURE_TARGET}")
        self._progress_bar.setValue(n)

    def resizeEvent(self, event):
        # Make sure the countdown overlay always covers the dialog
        self._countdown.setGeometry(self.rect())
        super().resizeEvent(event)
