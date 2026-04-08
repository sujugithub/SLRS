"""
Design system for SignBridge — dark editorial aesthetic, v2.

Palette  : deep charcoal / near-black (#08090d) + crisp white text +
           a single electric teal accent (#00e0c6).
Typography: JetBrains Mono / SF Mono for technical labels and stats,
            paired with Inter / SF Pro Display for body text.
Surfaces : flat, hairline 1px borders, subtle elevated tints.

This module is intentionally backwards compatible — every constant and
component name that the rest of the GUI imports is preserved, but the
underlying values and styling have been refreshed.
"""
from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtCore import (
    Qt, QEasingCurve, QPropertyAnimation, QRectF, QTimer, pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QCursor, QFont, QFontDatabase, QImage, QPainter, QPainterPath,
    QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect,
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

# ─── Color Palette ────────────────────────────────────────────────────────────
BG_DEEP      = "#08090d"    # page background
BG_BASE      = "#0b0d13"    # nav bars, hero strips
BG_SURFACE   = "#10131c"    # panel / card surfaces
BG_ELEVATED  = "#161a26"    # slightly elevated cards
BG_FLOAT     = "#1c2130"    # input backgrounds
BG_BORDER    = "#1f2538"    # used in lerp() — keep as hex
BG_ACTIVE    = "#212740"    # pressed state

# Electric teal — the single accent colour
ACCENT       = "#00e0c6"
ACCENT_HOVER = "#33ead4"
ACCENT_DARK  = "#00b69e"
ACCENT_TINT  = "#0a1f1c"

# Status colours
SUCCESS      = "#10d99a"
SUCCESS_TINT = "#0a1f17"
WARNING      = "#ffb24d"
DANGER       = "#ff5577"
DANGER_TINT  = "#1f0a10"
INFO         = "#5db4ff"

# Text — crisp cool white hierarchy
TEXT_TITLE   = "#f4f6fb"    # primary / headings
TEXT_PRIMARY = "#e1e5ef"    # body text
TEXT_SEC     = "#8a93a8"    # secondary / muted labels
TEXT_HINT    = "#525a72"    # disabled / placeholder

# Hairline border for use directly in QSS strings (not in lerp)
HAIR        = "rgba(244,246,251,0.07)"
HAIR_STRONG = "rgba(244,246,251,0.14)"


# ─── Typography ───────────────────────────────────────────────────────────────
# Best-available technical mono and modern sans, picked at runtime.
def _pick_font(candidates: list[str], fallback: str) -> str:
    """Return the first font from ``candidates`` that exists on this system."""
    families = set(QFontDatabase.families())
    for name in candidates:
        if name in families:
            return name
    return fallback


def _resolve_fonts() -> tuple[str, str]:
    mono = _pick_font(
        [
            "JetBrains Mono", "JetBrainsMono Nerd Font",
            "SF Mono", "Menlo", "Monaco", "Consolas",
            "Fira Code", "Source Code Pro",
        ],
        fallback="Courier New",
    )
    display = _pick_font(
        [
            "Inter", "SF Pro Display", "SF Pro Text",
            "Helvetica Neue", "Arial",
        ],
        fallback="Helvetica",
    )
    return mono, display


# These are populated lazily so QFontDatabase is queried after QApplication
# is constructed (otherwise families() can be empty on some platforms).
FONT_MONO: str = "Menlo"
FONT_DISPLAY: str = "Helvetica Neue"


def init_fonts() -> None:
    """Resolve the best system fonts.

    Must be called after ``QApplication`` is constructed. Safe to call
    multiple times.
    """
    global FONT_MONO, FONT_DISPLAY
    mono, display = _resolve_fonts()
    FONT_MONO    = mono
    FONT_DISPLAY = display


# ─── Color Utilities ──────────────────────────────────────────────────────────
def lerp(c1: str, c2: str, t: float) -> str:
    """Linear interpolation between two hex color strings."""
    def _p(c: str):
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r1, g1, b1 = _p(c1)
    r2, g2, b2 = _p(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )


def darken(c: str, amount: float = 0.15) -> str:
    return lerp(c, "#000000", amount)


def lighten(c: str, amount: float = 0.12) -> str:
    return lerp(c, "#ffffff", amount)


def confidence_color(value: float) -> str:
    """Map a 0..1 confidence to a colour gradient: red → amber → green → teal."""
    v = max(0.0, min(1.0, value))
    if v < 0.4:
        return lerp(DANGER, WARNING, v / 0.4)
    if v < 0.75:
        return lerp(WARNING, SUCCESS, (v - 0.4) / 0.35)
    return lerp(SUCCESS, ACCENT, (v - 0.75) / 0.25)


# ─── OpenCV → Qt ──────────────────────────────────────────────────────────────
def cv2_to_qpixmap(frame: np.ndarray | None) -> QPixmap | None:
    """Convert OpenCV BGR frame to QPixmap."""
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(bytes(rgb.data), w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ─── Animation Helper ─────────────────────────────────────────────────────────
def fade_in(widget: QWidget, duration: int = 220) -> QPropertyAnimation:
    """Fade-in animation via QGraphicsOpacityEffect."""
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start()
    return anim


# ─── Global Stylesheet ────────────────────────────────────────────────────────
def build_global_qss() -> str:
    return f"""
QWidget {{
    background-color: {BG_DEEP};
    color: {TEXT_PRIMARY};
    font-family: "{FONT_DISPLAY}", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}
QMainWindow {{ background-color: {BG_DEEP}; }}

QLineEdit {{
    background-color: {BG_FLOAT};
    border: 1px solid {HAIR};
    border-radius: 4px;
    padding: 10px 14px;
    color: {TEXT_TITLE};
    font-family: "{FONT_MONO}", "SF Mono", Menlo, monospace;
    font-size: 13px;
    selection-background-color: {ACCENT};
    selection-color: {BG_DEEP};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
    background-color: {BG_FLOAT};
}}

QProgressBar {{
    background-color: {HAIR};
    border: none;
    border-radius: 2px;
    text-align: center;
    color: transparent;
    height: 4px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 2px;
}}

QListWidget {{
    background-color: {BG_SURFACE};
    border: 1px solid {HAIR};
    border-radius: 4px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 10px 14px;
    border-radius: 3px;
    color: {TEXT_PRIMARY};
    font-size: 12px;
    font-family: "{FONT_MONO}", monospace;
}}
QListWidget::item:selected {{
    background-color: rgba(0,224,198,0.14);
    color: {TEXT_TITLE};
}}
QListWidget::item:hover:!selected {{ background-color: rgba(244,246,251,0.04); }}

QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BG_FLOAT};
    border-radius: 3px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {BG_ELEVATED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 6px; }}
QScrollBar::handle:horizontal {{
    background: {BG_FLOAT};
    border-radius: 3px;
    min-width: 28px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QRadioButton {{
    color: {TEXT_PRIMARY};
    spacing: 10px;
    font-size: 12px;
    font-family: "{FONT_DISPLAY}", sans-serif;
}}
QRadioButton::indicator {{
    width: 14px; height: 14px; border-radius: 7px;
    border: 1px solid {HAIR_STRONG}; background-color: {BG_ELEVATED};
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 10px;
    font-size: 12px;
    font-family: "{FONT_DISPLAY}", sans-serif;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid {HAIR_STRONG}; background-color: {BG_ELEVATED};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QScrollArea {{ border: none; background: transparent; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {BG_BORDER};
    background: {BG_BORDER};
    border: none;
    max-height: 1px;
}}
QToolTip {{
    background-color: {BG_ELEVATED};
    color: {TEXT_TITLE};
    border: 1px solid {HAIR_STRONG};
    border-radius: 3px;
    padding: 5px 9px;
    font-family: "{FONT_MONO}", monospace;
    font-size: 11px;
}}
QMessageBox {{
    background-color: {BG_SURFACE};
}}
QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
    font-family: "{FONT_DISPLAY}", sans-serif;
}}
"""


# Backwards-compatible attribute that other modules import. Updated by
# ``rebuild_global_qss()`` once fonts are resolved.
GLOBAL_QSS = build_global_qss()


def rebuild_global_qss() -> str:
    """Re-render GLOBAL_QSS using the latest font names."""
    global GLOBAL_QSS
    GLOBAL_QSS = build_global_qss()
    return GLOBAL_QSS


# ─── Components ───────────────────────────────────────────────────────────────

class GlassCard(QFrame):
    """Flat dark card with hairline border. Optional drop-shadow."""

    def __init__(self, parent=None, radius=6, bg=BG_SURFACE,
                 border_color=None, shadow=False):
        super().__init__(parent)
        self._bg = bg
        self._border = border_color or HAIR
        self._radius = radius
        self._apply_style()
        if shadow:
            eff = QGraphicsDropShadowEffect(self)
            eff.setBlurRadius(28)
            eff.setOffset(0.0, 4.0)
            eff.setColor(QColor(0, 0, 0, 90))
            self.setGraphicsEffect(eff)

    def _apply_style(self):
        self.setStyleSheet(
            f"QFrame {{ background-color: {self._bg}; "
            f"border: 1px solid {self._border}; "
            f"border-radius: {self._radius}px; }}"
            f"QLabel {{ background: transparent; border: none; }}"
            f"QProgressBar {{ border: none; }}"
        )

    def set_border_color(self, color: str):
        self._border = color
        self._apply_style()


class PillButton(QPushButton):
    """Filled primary button — solid accent background, uppercase mono label."""

    def __init__(self, text="", color=ACCENT, hover_color=None,
                 text_color=BG_DEEP, parent=None):
        super().__init__(text, parent)
        self._base  = color
        self._hover = hover_color or lighten(color, 0.12)
        self._txt   = text_color if color == ACCENT else TEXT_TITLE
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(34)
        self._paint(self._base)

    def _paint(self, bg: str):
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: {bg};"
            f" color: {self._txt};"
            f" border: 1px solid {bg};"
            f" border-radius: 4px;"
            f" padding: 8px 22px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-size: 11px;"
            f" font-weight: 600;"
            f" letter-spacing: 1.4px;"
            f" text-transform: uppercase;"
            f" }}"
            f"QPushButton:disabled {{"
            f" background-color: {BG_BORDER};"
            f" color: {TEXT_HINT};"
            f" border-color: {BG_BORDER};"
            f" }}"
        )

    def enterEvent(self, e):
        self._paint(self._hover)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._paint(self._base)
        super().leaveEvent(e)


class OutlineButton(QPushButton):
    """Flat outline button — no fill, hairline border, uppercase mono label."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(32)
        self._paint_normal()

    def _paint_normal(self):
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: transparent;"
            f" color: {TEXT_PRIMARY};"
            f" border: 1px solid {HAIR_STRONG};"
            f" border-radius: 4px;"
            f" padding: 7px 16px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-size: 10px;"
            f" font-weight: 600;"
            f" letter-spacing: 1.2px;"
            f" }}"
            f"QPushButton:disabled {{"
            f" color: {TEXT_HINT};"
            f" border-color: {HAIR};"
            f" }}"
        )

    def _paint_hover(self):
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: rgba(0,224,198,0.08);"
            f" color: {ACCENT};"
            f" border: 1px solid {ACCENT};"
            f" border-radius: 4px;"
            f" padding: 7px 16px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-size: 10px;"
            f" font-weight: 600;"
            f" letter-spacing: 1.2px;"
            f" }}"
        )

    def enterEvent(self, e):
        self._paint_hover()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._paint_normal()
        super().leaveEvent(e)


class DangerButton(OutlineButton):
    """Outline button with red tint."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._paint_danger_normal()

    def _paint_danger_normal(self):
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: transparent;"
            f" color: {DANGER};"
            f" border: 1px solid rgba(255,85,119,0.40);"
            f" border-radius: 4px;"
            f" padding: 7px 16px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-size: 10px;"
            f" font-weight: 600;"
            f" letter-spacing: 1.2px;"
            f" }}"
            f"QPushButton:disabled {{"
            f" color: {TEXT_HINT};"
            f" }}"
        )

    def _paint_danger_hover(self):
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: rgba(255,85,119,0.14);"
            f" color: #ff7790;"
            f" border: 1px solid {DANGER};"
            f" border-radius: 4px;"
            f" padding: 7px 16px;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" font-size: 10px;"
            f" font-weight: 600;"
            f" letter-spacing: 1.2px;"
            f" }}"
        )

    def enterEvent(self, e):
        self._paint_danger_hover()
        super(OutlineButton, self).enterEvent(e)

    def leaveEvent(self, e):
        self._paint_danger_normal()
        super(OutlineButton, self).leaveEvent(e)


class IconButton(QPushButton):
    """Square borderless icon button — used for top-bar hamburger / close."""

    def __init__(self, glyph: str = "", size: int = 36, parent=None):
        super().__init__(glyph, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedSize(size, size)
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: transparent;"
            f" color: {TEXT_SEC};"
            f" border: 1px solid transparent;"
            f" border-radius: 6px;"
            f" font-size: 17px;"
            f" }}"
            f"QPushButton:hover {{"
            f" background-color: {BG_ELEVATED};"
            f" border-color: {HAIR_STRONG};"
            f" color: {TEXT_TITLE};"
            f" }}"
            f"QPushButton:pressed {{"
            f" background-color: {BG_ACTIVE};"
            f" }}"
        )


class PulsingDot(QLabel):
    """Small dot that pulses between active and dim every ~1100 ms."""

    def __init__(self, on_color=SUCCESS, off_color="#0a3a2a", size=8, parent=None):
        super().__init__(parent)
        self._on    = on_color
        self._off   = off_color
        self._size  = size
        self._state = True
        self.setFixedSize(size + 4, size + 4)
        r = (size + 4) // 2
        self._r = r
        self._update()
        self._timer = QTimer(self)
        self._timer.setInterval(1100)
        self._timer.timeout.connect(self._toggle)
        self._timer.start()

    def _toggle(self):
        self._state = not self._state
        self._update()

    def _update(self):
        c = self._on if self._state else self._off
        self.setStyleSheet(
            f"QLabel {{ background-color: {c};"
            f" border-radius: {self._r}px; border: none; }}"
        )


class StatusDot(QLabel):
    """Tiny coloured dot status indicator (no animation)."""

    def __init__(self, color=TEXT_HINT, size=8, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size + 4, size + 4)
        self.set_color(color)

    def set_color(self, color: str):
        r = (self._size + 4) // 2
        self.setStyleSheet(
            f"QLabel {{ background-color: {color};"
            f" border-radius: {r}px; border: none; }}"
        )


class SectionLabel(QLabel):
    """Small ALL-CAPS monospace section header."""

    def __init__(self, text="", parent=None):
        super().__init__(text.upper(), parent)
        self.setStyleSheet(
            f"QLabel {{ color: {TEXT_HINT};"
            f" font-size: 9px;"
            f" font-weight: 700;"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 2px;"
            f" background: transparent; border: none; }}"
        )


class HSep(QFrame):
    """Thin horizontal separator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background-color: {BG_BORDER}; border: none;")


class CardButton(QFrame):
    """Large animated tile — kept for backwards compatibility."""

    clicked = pyqtSignal()

    def __init__(self, icon="", title="", subtitle="", accent=ACCENT, parent=None):
        super().__init__(parent)
        self._accent = accent
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumSize(220, 170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(10)
        lay.setContentsMargins(28, 24, 28, 24)

        self._icon_lbl = QLabel(icon)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet(
            f"font-size: 30px; color: {accent};"
            f" background: transparent; border: none;"
        )

        self._title_lbl = QLabel(title)
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {TEXT_TITLE};"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            f" letter-spacing: 0.5px;"
            f" background: transparent; border: none;"
        )

        self._sub_lbl = QLabel(subtitle)
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_lbl.setWordWrap(True)
        self._sub_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_SEC};"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 1px;"
            f" background: transparent; border: none;"
        )

        lay.addWidget(self._icon_lbl)
        lay.addWidget(self._title_lbl)
        lay.addWidget(self._sub_lbl)

        self._set_normal()

    def _set_normal(self):
        self.setStyleSheet(
            f"QFrame {{ background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR}; border-radius: 6px; }}"
            f"QLabel {{ background: transparent; border: none; }}"
        )

    def _set_hovered(self):
        self.setStyleSheet(
            f"QFrame {{ background-color: {BG_ELEVATED};"
            f" border: 1px solid {HAIR_STRONG}; border-radius: 6px; }}"
            f"QLabel {{ background: transparent; border: none; }}"
        )

    def _set_pressed(self):
        self.setStyleSheet(
            f"QFrame {{ background-color: {BG_ACTIVE};"
            f" border: 1px solid {ACCENT}; border-radius: 6px; }}"
            f"QLabel {{ background: transparent; border: none; }}"
        )

    def enterEvent(self, e): self._set_hovered(); super().enterEvent(e)
    def leaveEvent(self, e): self._set_normal(); super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._set_pressed()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._set_hovered()
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class CameraFrame(QLabel):
    """QLabel that draws a camera feed inside a rounded frame with LIVE label."""

    def __init__(self, w=560, h=420, radius=6, parent=None):
        super().__init__(parent)
        self._radius = radius
        self._pix: QPixmap | None = None
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-radius: {radius}px;"
        )

    def set_frame(self, pix: QPixmap | None):
        self._pix = pix
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._radius > 0:
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), self._radius, self._radius)
            p.setClipPath(path)

        if self._pix:
            scaled = self._pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)

            # LIVE indicator with pulse dot — top-left
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(ACCENT))
            p.drawEllipse(14, 14, 7, 7)
            p.setPen(QColor(244, 246, 251, 230))
            font = QFont(FONT_MONO, 9)
            font.setBold(True)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.4)
            p.setFont(font)
            p.drawText(28, 22, "LIVE")
        else:
            p.fillRect(self.rect(), QColor(BG_BASE))
            p.setPen(QColor(TEXT_HINT))
            font = QFont(FONT_MONO, 11)
            p.setFont(font)
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "CAMERA OFFLINE\nfeed will appear when active",
            )
        p.end()


class AnimatedConfidenceBar(QWidget):
    """Custom-painted confidence bar that animates fill % and colour."""

    def __init__(self, parent=None, height: int = 8):
        super().__init__(parent)
        self._target = 0.0
        self._value  = 0.0
        self.setMinimumHeight(height)
        self.setFixedHeight(height)
        self._anim = QPropertyAnimation(self, b"value", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def get_value(self) -> float:
        return self._value

    def set_value(self, v: float) -> None:
        self._value = max(0.0, min(1.0, float(v)))
        self.update()

    value = pyqtProperty(float, fget=get_value, fset=set_value)

    def set_confidence(self, v: float) -> None:
        target = max(0.0, min(1.0, float(v)))
        self._target = target
        self._anim.stop()
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(target)
        self._anim.start()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        radius = rect.height() / 2

        # Track
        track_path = QPainterPath()
        track_path.addRoundedRect(QRectF(rect), radius, radius)
        p.fillPath(track_path, QColor(28, 33, 48))

        # Fill
        if self._value > 0:
            fill_w = int(rect.width() * self._value)
            fill_rect = QRectF(0, 0, fill_w, rect.height())
            fill_path = QPainterPath()
            fill_path.addRoundedRect(fill_rect, radius, radius)
            color = QColor(confidence_color(self._value))
            p.fillPath(fill_path, color)
        p.end()
