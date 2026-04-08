"""
Full-screen 3 → 2 → 1 → GO countdown overlay for SignBridge.

Used by the Train and Add-samples flows immediately before each recording
starts. Displays one number per second on a semi-transparent dark backdrop,
emits ``finished`` when the GO step elapses, and can be canceled at any time
with the Escape key (which emits ``canceled`` instead).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QWidget

from gui.design import TEXT_TITLE, ACCENT, FONT_DISPLAY, FONT_MONO


class CountdownOverlay(QWidget):
    """Reusable full-screen 3-2-1-GO countdown.

    Signals:
        finished: emitted after the GO step elapses normally.
        canceled: emitted when the user presses Escape.
    """

    finished = pyqtSignal()
    canceled = pyqtSignal()

    _SEQUENCE = ("3", "2", "1", "GO")
    _STEP_MS  = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._step  = 0
        self._text  = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self.hide()

    # ─── Public API ───────────────────────────────────────────────────────────
    def start(self) -> None:
        """Begin the 3-2-1-GO sequence."""
        self._step = 0
        self._text = self._SEQUENCE[0]
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()
        self.setFocus()
        self.update()
        self._timer.start(self._STEP_MS)

    def cancel(self) -> None:
        """Stop the sequence and emit ``canceled``."""
        self._timer.stop()
        self.hide()
        self.canceled.emit()

    # ─── Internals ────────────────────────────────────────────────────────────
    def _advance(self) -> None:
        self._step += 1
        if self._step >= len(self._SEQUENCE):
            self.hide()
            self.finished.emit()
            return
        self._text = self._SEQUENCE[self._step]
        self.update()
        self._timer.start(self._STEP_MS)

    # ─── Qt overrides ─────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent very-dark backdrop
        p.fillRect(self.rect(), QColor(8, 9, 13, 235))

        # Eyebrow above the number
        eyebrow_font = QFont(FONT_MONO, 11)
        eyebrow_font.setBold(True)
        eyebrow_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4)
        p.setFont(eyebrow_font)
        p.setPen(QColor(0, 224, 198, 220))
        eyebrow_rect = self.rect().adjusted(0, 0, 0, 0)
        eyebrow_rect.setHeight(self.rect().height() // 2 - 130)
        p.drawText(
            eyebrow_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            "RECORDING IN",
        )

        # Big numeral / "GO"
        font = QFont(FONT_DISPLAY, 240, QFont.Weight.Black)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -6)
        p.setFont(font)
        color = QColor(ACCENT) if self._text == "GO" else QColor(TEXT_TITLE)
        p.setPen(color)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)

        # Hint text
        p.setPen(QColor(138, 147, 168, 220))
        hint_font = QFont(FONT_MONO, 10)
        hint_font.setBold(True)
        hint_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        p.setFont(hint_font)
        hint = "PRESS ESC TO CANCEL"
        rect = self.rect().adjusted(0, 0, 0, -50)
        p.drawText(
            rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            hint,
        )

        p.end()

    def resizeEvent(self, event):
        # Keep covering the parent fully if it resizes mid-countdown
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)
