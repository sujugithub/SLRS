"""
Settings dialog for SignBridge.

Six controls — confidence threshold, TTS on/off, TTS voice, camera index,
MediaPipe complexity, and prediction smoothing window — wired to a single
Save button. Emits ``settings_changed(dict)`` so MainWindow can apply the
new values live (camera restart, smoother resize, threshold/TTS update).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget,
)

from gui.design import (
    ACCENT, BG_BASE, BG_DEEP, BG_FLOAT, BG_SURFACE, HAIR,
    SUCCESS, TEXT_HINT, TEXT_PRIMARY, TEXT_SEC, TEXT_TITLE,
    GlassCard, HSep, PillButton, OutlineButton, SectionLabel,
)


def _value_label(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"font-size: 12px; font-weight: 600; color: {TEXT_TITLE};"
        f" font-family: 'Courier New', monospace;"
        " background: transparent; border: none;"
    )
    return lbl


def _hint_label(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"font-size: 10px; color: {TEXT_HINT};"
        f" font-family: 'Courier New', monospace;"
        " background: transparent; border: none;"
    )
    return lbl


class SettingsDialog(QWidget):
    """Settings page added to the MainWindow stack."""

    settings_changed = pyqtSignal(dict)
    back_requested   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ─── UI Construction ──────────────────────────────────────────────────────
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
        back_btn.clicked.connect(self.back_requested.emit)
        nav_lay.addWidget(back_btn)

        title_lbl = QLabel("Settings")
        title_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {TEXT_TITLE};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none;"
        )
        nav_lay.addWidget(title_lbl, 1)
        root.addWidget(nav)

        # Body container
        body = QWidget()
        body.setStyleSheet(f"background-color: {BG_DEEP};")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(40, 28, 40, 28)
        body_lay.setSpacing(16)
        root.addWidget(body, 1)

        card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(28, 24, 28, 24)
        card_lay.setSpacing(20)
        body_lay.addWidget(card, 0, Qt.AlignmentFlag.AlignTop)

        # 1 — Confidence threshold
        card_lay.addWidget(SectionLabel("Confidence Threshold"))
        ct_row = QHBoxLayout()
        self._ct_slider = QSlider(Qt.Orientation.Horizontal)
        self._ct_slider.setRange(0, 100)
        self._ct_slider.setValue(60)
        self._ct_slider.setStyleSheet(self._slider_qss())
        self._ct_slider.valueChanged.connect(self._on_ct_changed)
        ct_row.addWidget(self._ct_slider, 1)
        self._ct_value = _value_label("0.60")
        self._ct_value.setFixedWidth(48)
        ct_row.addWidget(self._ct_value)
        card_lay.addLayout(ct_row)
        card_lay.addWidget(_hint_label(
            "Predictions below this confidence are ignored "
            "(no TTS, no auto-add)."
        ))

        card_lay.addWidget(HSep())

        # 2 — TTS enabled
        tts_row = QHBoxLayout()
        self._tts_cb = QCheckBox("Speak detected signs")
        self._tts_cb.setChecked(True)
        tts_row.addWidget(self._tts_cb)
        tts_row.addStretch(1)
        card_lay.addLayout(tts_row)

        # 3 — TTS voice
        card_lay.addWidget(SectionLabel("TTS Voice"))
        self._voice_combo = QComboBox()
        self._voice_combo.setStyleSheet(self._combo_qss())
        self._voice_combo.addItem("System default", "")
        card_lay.addWidget(self._voice_combo)

        card_lay.addWidget(HSep())

        # 4 — Camera index
        card_lay.addWidget(SectionLabel("Camera"))
        self._cam_combo = QComboBox()
        self._cam_combo.setStyleSheet(self._combo_qss())
        card_lay.addWidget(self._cam_combo)
        card_lay.addWidget(_hint_label(
            "Detected video inputs. Switching takes effect immediately."
        ))

        card_lay.addWidget(HSep())

        # 5 — MediaPipe complexity
        card_lay.addWidget(SectionLabel("MediaPipe Complexity"))
        self._mp_combo = QComboBox()
        self._mp_combo.setStyleSheet(self._combo_qss())
        self._mp_combo.addItem("Light  (fastest)", 0)
        self._mp_combo.addItem("Full  (balanced)", 1)
        self._mp_combo.addItem("Heavy (most accurate)", 2)
        card_lay.addWidget(self._mp_combo)
        card_lay.addWidget(_hint_label(
            "Falls back to Light automatically if higher-complexity "
            ".task files are not bundled."
        ))

        card_lay.addWidget(HSep())

        # 6 — Smoothing window
        card_lay.addWidget(SectionLabel("Prediction Smoothing"))
        sw_row = QHBoxLayout()
        self._sw_slider = QSlider(Qt.Orientation.Horizontal)
        self._sw_slider.setRange(1, 15)
        self._sw_slider.setValue(5)
        self._sw_slider.setStyleSheet(self._slider_qss())
        self._sw_slider.valueChanged.connect(self._on_sw_changed)
        sw_row.addWidget(self._sw_slider, 1)
        self._sw_value = _value_label("5")
        self._sw_value.setFixedWidth(48)
        sw_row.addWidget(self._sw_value)
        card_lay.addLayout(sw_row)
        card_lay.addWidget(_hint_label(
            "Frames of confidence-weighted majority vote. "
            "1 = no smoothing."
        ))

        card_lay.addSpacing(8)

        # Save button
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self._save_btn = PillButton("SAVE", color=ACCENT)
        self._save_btn.clicked.connect(self._on_save_clicked)
        save_row.addWidget(self._save_btn)
        card_lay.addLayout(save_row)

        body_lay.addStretch(1)

        # Populate dynamic options
        self._populate_voice_combo()
        self._populate_camera_combo()

    # ─── Slider/combo helpers ─────────────────────────────────────────────────
    def _slider_qss(self) -> str:
        return (
            f"QSlider::groove:horizontal {{ background: {HAIR}; height: 4px;"
            f" border-radius: 0; }}"
            f"QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px;"
            f" margin: -6px 0; border-radius: 7px; }}"
            f"QSlider::sub-page:horizontal {{ background: {ACCENT}; }}"
        )

    def _combo_qss(self) -> str:
        return (
            f"QComboBox {{ background-color: {BG_FLOAT}; color: {TEXT_PRIMARY};"
            f" border: 1px solid {HAIR}; border-radius: 0; padding: 6px 10px;"
            f" font-family: 'Courier New', monospace; font-size: 12px; }}"
            f"QComboBox::drop-down {{ border: none; width: 22px; }}"
            f"QComboBox QAbstractItemView {{ background-color: {BG_FLOAT};"
            f" color: {TEXT_PRIMARY}; selection-background-color: {ACCENT};"
            f" border: 1px solid {HAIR}; outline: none; }}"
        )

    # ─── Dynamic population ───────────────────────────────────────────────────
    def _populate_voice_combo(self) -> None:
        """Try to import TTSSpeaker and ask it for voices.

        Failure here is non-fatal — the dropdown still has the
        "System default" entry inserted in _build_ui.
        """
        try:
            from core.tts_speaker import TTSSpeaker
            speaker = TTSSpeaker()
            voices = speaker.list_voices()
            speaker.stop()
        except Exception as exc:
            print(f"[SettingsDialog] voice probe failed: {exc}")
            return

        # Replace the placeholder default entry with the real list
        self._voice_combo.clear()
        for voice_id, label in voices:
            self._voice_combo.addItem(label, voice_id)
        if self._voice_combo.count() == 0:
            self._voice_combo.addItem("System default", "")

    def _populate_camera_combo(self) -> None:
        """Probe indices 0-5 and list any that open."""
        try:
            import cv2
        except Exception:
            self._cam_combo.addItem("Camera 0", 0)
            return

        found = False
        for idx in range(6):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self._cam_combo.addItem(f"Camera {idx}", idx)
                found = True
            cap.release()
        if not found:
            self._cam_combo.addItem("Camera 0  (no camera found)", 0)

    # ─── Slider callbacks ─────────────────────────────────────────────────────
    def _on_ct_changed(self, v: int) -> None:
        self._ct_value.setText(f"{v / 100:.2f}")

    def _on_sw_changed(self, v: int) -> None:
        self._sw_value.setText(str(v))

    # ─── Public API ───────────────────────────────────────────────────────────
    def load_into_ui(self, settings: dict) -> None:
        """Populate every control from the supplied settings dict."""
        ct = float(settings.get("confidence_threshold", 0.6))
        self._ct_slider.setValue(int(round(ct * 100)))

        self._tts_cb.setChecked(bool(settings.get("tts_enabled", True)))

        voice = settings.get("tts_voice", "")
        idx = self._voice_combo.findData(voice)
        if idx >= 0:
            self._voice_combo.setCurrentIndex(idx)
        else:
            self._voice_combo.setCurrentIndex(0)

        cam_idx = int(settings.get("camera_index", 0))
        idx = self._cam_combo.findData(cam_idx)
        if idx >= 0:
            self._cam_combo.setCurrentIndex(idx)
        else:
            self._cam_combo.addItem(f"Camera {cam_idx}", cam_idx)
            self._cam_combo.setCurrentIndex(self._cam_combo.count() - 1)

        mp = int(settings.get("mediapipe_complexity", 0))
        idx = self._mp_combo.findData(mp)
        if idx >= 0:
            self._mp_combo.setCurrentIndex(idx)

        sw = int(settings.get("smoothing_window", 5))
        self._sw_slider.setValue(max(1, min(15, sw)))

    # ─── Save ─────────────────────────────────────────────────────────────────
    def _on_save_clicked(self) -> None:
        new_settings = {
            "confidence_threshold": self._ct_slider.value() / 100.0,
            "tts_enabled":          self._tts_cb.isChecked(),
            "tts_voice":            self._voice_combo.currentData() or "",
            "camera_index":         int(self._cam_combo.currentData() or 0),
            "mediapipe_complexity": int(self._mp_combo.currentData() or 0),
            "smoothing_window":     int(self._sw_slider.value()),
        }
        self.settings_changed.emit(new_settings)
