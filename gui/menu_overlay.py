"""
Menu drawer for SignBridge.

A full-window modal/drawer overlay that slides in from the right when the
hamburger button is pressed. Contains four pages — Train, View, Retrain,
Settings — accessed from a sidebar list. The sub-screen pages are hosted
inside the drawer's stacked widget so navigation feels contained rather
than full-page swaps.

The legacy ``HamburgerMenu`` popup name is also exported for backwards
compatibility, but the recommended class is ``MenuDrawer``.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import (
    Qt, QEasingCurve, QPropertyAnimation, QPoint, QRect, pyqtSignal,
)
from PyQt6.QtGui import QColor, QCursor, QPainter
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

from gui.design import (
    ACCENT, BG_BASE, BG_DEEP, BG_ELEVATED, BG_SURFACE, HAIR, HAIR_STRONG,
    TEXT_HINT, TEXT_PRIMARY, TEXT_SEC, TEXT_TITLE,
    FONT_DISPLAY, FONT_MONO, IconButton, SectionLabel,
)


# Menu items: (key, label, glyph, blurb)
_ITEMS = (
    ("train",    "Train",     "●",
     "Record samples for a new sign or phrase"),
    ("view",     "Library",   "▤",
     "Browse and manage trained signs"),
    ("retrain",  "Retrain",   "↻",
     "Retrain the model on every captured sample"),
    ("settings", "Settings",  "⚙",
     "Confidence threshold, voice, camera, smoothing"),
)


class _SidebarItem(QPushButton):
    """One row in the drawer sidebar."""

    def __init__(self, key: str, label: str, glyph: str, parent=None):
        super().__init__(parent)
        self.key   = key
        self._lbl  = label
        self._glyph = glyph
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setCheckable(True)
        self.setMinimumHeight(48)
        self.setText(f"  {glyph}     {label.upper()}")
        self._apply()

    def _apply(self):
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: transparent;"
            f" color: {TEXT_SEC};"
            f" border: none;"
            f" border-left: 2px solid transparent;"
            f" padding: 12px 22px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-size: 11px;"
            f" font-weight: 700;"
            f" letter-spacing: 1.5px;"
            f" text-align: left;"
            f" }}"
            f"QPushButton:hover {{"
            f" background-color: rgba(244,246,251,0.04);"
            f" color: {TEXT_TITLE};"
            f" }}"
            f"QPushButton:checked {{"
            f" background-color: rgba(0,224,198,0.08);"
            f" color: {ACCENT};"
            f" border-left: 2px solid {ACCENT};"
            f" }}"
        )


class _Backdrop(QWidget):
    """Semi-transparent dark backdrop behind the drawer panel."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(8, 9, 13, 170))
        p.end()

    def mousePressEvent(self, _e):
        self.clicked.emit()


class MenuDrawer(QWidget):
    """Full-window modal drawer that slides in from the right.

    Hosts a sidebar of menu items + a stacked widget of sub-screens.
    Sub-screens are added via ``add_screen(key, widget)`` and selected
    via ``select(key)``. Emits ``item_selected(key)`` when the user
    chooses an item, and ``closed()`` when the drawer is dismissed.
    """

    item_selected = pyqtSignal(str)
    closed        = pyqtSignal()

    DRAWER_WIDTH_FRAC = 0.78
    DRAWER_MIN_WIDTH  = 760
    SIDEBAR_WIDTH     = 220
    ANIM_MS           = 220

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setVisible(False)

        # Backdrop fills the whole parent
        self._backdrop = _Backdrop(self)
        self._backdrop.clicked.connect(self.close_drawer)

        # The actual sliding panel
        self._panel = QFrame(self)
        self._panel.setStyleSheet(
            f"QFrame {{ background-color: {BG_BASE};"
            f" border: none;"
            f" border-left: 1px solid {HAIR_STRONG}; }}"
        )

        panel_lay = QHBoxLayout(self._panel)
        panel_lay.setContentsMargins(0, 0, 0, 0)
        panel_lay.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setFixedWidth(self.SIDEBAR_WIDTH)
        sidebar.setStyleSheet(
            f"QFrame {{ background-color: {BG_DEEP};"
            f" border: none;"
            f" border-right: 1px solid {HAIR}; }}"
        )
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(0, 28, 0, 24)
        sb_lay.setSpacing(0)

        # Sidebar brand
        brand_wrap = QWidget()
        brand_lay = QVBoxLayout(brand_wrap)
        brand_lay.setContentsMargins(22, 0, 22, 24)
        brand_lay.setSpacing(2)
        brand_eyebrow = QLabel("MENU")
        brand_eyebrow.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 9px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 2.5px; font-weight: 700;"
            f" background: transparent; border: none;"
        )
        brand_title = QLabel("SignBridge")
        brand_title.setStyleSheet(
            f"color: {TEXT_TITLE}; font-size: 22px;"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            f" font-weight: 800; letter-spacing: -0.5px;"
            f" background: transparent; border: none;"
        )
        brand_lay.addWidget(brand_eyebrow)
        brand_lay.addWidget(brand_title)
        sb_lay.addWidget(brand_wrap)

        self._items: dict[str, _SidebarItem] = {}
        for key, label, glyph, _blurb in _ITEMS:
            item = _SidebarItem(key, label, glyph, sidebar)
            item.clicked.connect(lambda _checked=False, k=key: self._on_item_click(k))
            sb_lay.addWidget(item)
            self._items[key] = item

        sb_lay.addStretch(1)

        # Footer hint
        footer = QLabel("ESC TO CLOSE")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 9px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 2px; font-weight: 600;"
            f" background: transparent; border: none;"
            f" padding-top: 12px;"
        )
        sb_lay.addWidget(footer)

        panel_lay.addWidget(sidebar)

        # ── Content area ─────────────────────────────────────────────────
        content_wrap = QWidget()
        content_wrap.setStyleSheet(f"background-color: {BG_BASE};")
        content_lay = QVBoxLayout(content_wrap)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(0)

        # Header bar with title + close button
        header = QWidget()
        header.setFixedHeight(58)
        header.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-bottom: 1px solid {HAIR};"
        )
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(28, 0, 16, 0)
        header_lay.setSpacing(12)

        self._eyebrow = QLabel("SECTION")
        self._eyebrow.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 9px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 2.5px; font-weight: 700;"
            f" background: transparent; border: none;"
        )
        self._title_lbl = QLabel("Train")
        self._title_lbl.setStyleSheet(
            f"color: {TEXT_TITLE}; font-size: 16px;"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            f" font-weight: 700; letter-spacing: -0.2px;"
            f" background: transparent; border: none;"
        )
        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(0)
        title_col.addWidget(self._eyebrow)
        title_col.addWidget(self._title_lbl)
        header_lay.addLayout(title_col)
        header_lay.addStretch(1)

        close_btn = IconButton("✕", size=36)
        close_btn.clicked.connect(self.close_drawer)
        header_lay.addWidget(close_btn)

        content_lay.addWidget(header)

        # Stack of sub-screens
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background-color: {BG_DEEP};")
        content_lay.addWidget(self._stack, 1)

        panel_lay.addWidget(content_wrap, 1)

        self._screens: dict[str, QWidget] = {}
        self._current_key: Optional[str] = None

        # Slide animation
        self._anim = QPropertyAnimation(self._panel, b"geometry", self)
        self._anim.setDuration(self.ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Public API ───────────────────────────────────────────────────────────
    def add_screen(self, key: str, widget: QWidget) -> None:
        """Register a sub-screen widget for the given menu key."""
        self._screens[key] = widget
        self._stack.addWidget(widget)

    def open_drawer(self, initial_key: str = "train") -> None:
        """Show + slide-in animate, selecting ``initial_key``."""
        if not self.parent():
            return
        parent_rect = self.parent().rect()
        self.setGeometry(parent_rect)
        self.setVisible(True)
        self.raise_()

        self._backdrop.setGeometry(0, 0, parent_rect.width(), parent_rect.height())

        panel_w = max(self.DRAWER_MIN_WIDTH,
                      int(parent_rect.width() * self.DRAWER_WIDTH_FRAC))
        panel_w = min(panel_w, parent_rect.width())
        panel_h = parent_rect.height()
        end = QRect(parent_rect.width() - panel_w, 0, panel_w, panel_h)
        start = QRect(parent_rect.width(), 0, panel_w, panel_h)

        self._panel.setGeometry(start)
        self._anim.stop()
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        # Clear any prior finish handler so close handlers don't fire on open
        try:
            self._anim.finished.disconnect()
        except TypeError:
            pass
        self._anim.start()

        self.select(initial_key)
        self.setFocus()

    def close_drawer(self) -> None:
        """Slide-out animate then hide and emit ``closed``."""
        if not self.isVisible() or not self.parent():
            return
        parent_rect = self.parent().rect()
        cur = self._panel.geometry()
        end = QRect(parent_rect.width(), cur.y(), cur.width(), cur.height())
        self._anim.stop()
        self._anim.setStartValue(cur)
        self._anim.setEndValue(end)
        try:
            self._anim.finished.disconnect()
        except TypeError:
            pass
        self._anim.finished.connect(self._after_close)
        self._anim.start()

    def _after_close(self) -> None:
        self.setVisible(False)
        try:
            self._anim.finished.disconnect()
        except TypeError:
            pass
        self.closed.emit()

    def select(self, key: str) -> None:
        """Switch to the sub-screen registered under ``key``."""
        if key not in self._screens:
            return
        self._current_key = key
        for k, btn in self._items.items():
            btn.setChecked(k == key)
        # Title
        for ikey, label, _g, blurb in _ITEMS:
            if ikey == key:
                self._title_lbl.setText(label)
                self._eyebrow.setText(blurb.upper())
                break
        self._stack.setCurrentWidget(self._screens[key])
        self.item_selected.emit(key)

    # ── Backwards-compat shim for old HamburgerMenu callsites ────────────────
    def show_at(self, _anchor_global_pos: QPoint) -> None:
        """Compatibility shim — open drawer at default page."""
        self.open_drawer()

    # ── Qt overrides ─────────────────────────────────────────────────────────
    def resizeEvent(self, event):
        if self.parent() is not None and self.isVisible():
            parent_rect = self.parent().rect()
            self.setGeometry(parent_rect)
            self._backdrop.setGeometry(
                0, 0, parent_rect.width(), parent_rect.height(),
            )
            panel_w = max(self.DRAWER_MIN_WIDTH,
                          int(parent_rect.width() * self.DRAWER_WIDTH_FRAC))
            panel_w = min(panel_w, parent_rect.width())
            self._panel.setGeometry(
                parent_rect.width() - panel_w, 0,
                panel_w, parent_rect.height(),
            )
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close_drawer()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_item_click(self, key: str) -> None:
        self.select(key)


# Backwards-compat alias — main.py used to import this name.
HamburgerMenu = MenuDrawer
