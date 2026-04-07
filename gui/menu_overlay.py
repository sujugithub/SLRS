"""
Hamburger menu popup for SignBridge.

A small frameless popup anchored under the ☰ button in the main window's
top bar. Contains four items — Train · View · Retrain · Settings — and
emits ``item_selected(key)`` when one is clicked. Auto-dismisses on
focus loss or Escape.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QFrame, QPushButton, QVBoxLayout

from gui.design import (
    BG_ELEVATED, HAIR, TEXT_PRIMARY, TEXT_TITLE, TEXT_HINT,
)


class _MenuItem(QPushButton):
    """One row in the hamburger popup."""

    def __init__(self, label: str, parent=None):
        super().__init__(label, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFlat(True)
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: transparent;"
            f" color: {TEXT_PRIMARY};"
            f" border: none;"
            f" padding: 10px 18px;"
            f" font-family: 'Courier New', monospace;"
            f" font-size: 11px;"
            f" letter-spacing: 1.5px;"
            f" text-align: left;"
            f"}}"
            f"QPushButton:hover {{"
            f" background-color: rgba(232,228,220,0.06);"
            f" color: {TEXT_TITLE};"
            f"}}"
        )


class HamburgerMenu(QFrame):
    """Frameless popup with four pill-styled menu items.

    Emits :pyattr:`item_selected` with one of:
    ``"train"``, ``"view"``, ``"retrain"``, ``"settings"``.
    """

    item_selected = pyqtSignal(str)

    _ITEMS = (
        ("train",    "TRAIN"),
        ("view",     "VIEW"),
        ("retrain",  "RETRAIN"),
        ("settings", "SETTINGS"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedWidth(200)
        self.setStyleSheet(
            f"HamburgerMenu {{"
            f" background-color: {BG_ELEVATED};"
            f" border: 1px solid {HAIR};"
            f"}}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(0)
        for key, label in self._ITEMS:
            btn = _MenuItem(label, self)
            btn.clicked.connect(lambda _checked=False, k=key: self._on_click(k))
            lay.addWidget(btn)

    # ─── Public API ───────────────────────────────────────────────────────────
    def show_at(self, anchor_global_pos: QPoint) -> None:
        """Show the popup with its top-left at ``anchor_global_pos``.

        Anchors are clamped to the screen if they would overflow.
        """
        self.adjustSize()
        screen = self.screen() or self.parent().screen() if self.parent() else None
        x, y = anchor_global_pos.x(), anchor_global_pos.y()
        if screen is not None:
            geo = screen.availableGeometry()
            x = min(x, geo.right() - self.width() - 4)
            y = min(y, geo.bottom() - self.height() - 4)
            x = max(x, geo.left() + 4)
            y = max(y, geo.top() + 4)
        self.move(x, y)
        self.show()
        self.raise_()
        self.setFocus()

    # ─── Internals ────────────────────────────────────────────────────────────
    def _on_click(self, key: str) -> None:
        self.hide()
        self.item_selected.emit(key)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)
