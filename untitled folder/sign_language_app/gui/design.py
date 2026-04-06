"""
Design system for SignBridge — dark editorial aesthetic.

Palette  : near-black (#0a0a09) + warm cream text + muted green accents only
Typography: "Courier New" / monospace for labels & data; serif for headings
Surfaces : flat, zero drop-shadows, hairline 1px rgba borders
"""
from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect,
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

# ─── Color Palette ────────────────────────────────────────────────────────────
BG_DEEP      = "#0a0a09"    # near-black page background
BG_BASE      = "#0d0d0c"    # nav bars, hero strips
BG_SURFACE   = "#111110"    # panel / card surfaces
BG_ELEVATED  = "#141413"    # slightly elevated cards
BG_FLOAT     = "#181817"    # input backgrounds
BG_BORDER    = "#1f1f1d"    # used in lerp() for camera glow — keep as hex
BG_ACTIVE    = "#1e1e1c"    # pressed state

# Green accent — status indicators ONLY, never decorative fill
ACCENT       = "#4a7c59"
ACCENT_HOVER = "#5a8c69"
ACCENT_DARK  = "#3a6c49"
ACCENT_TINT  = "#0d1a11"

SUCCESS      = "#7bb08a"    # lighter green for active/ready states
SUCCESS_TINT = "#0d1a11"
WARNING      = "#b8924a"    # amber, kept muted
DANGER       = "#8a4040"    # red, kept muted
DANGER_TINT  = "#1a0a0a"
INFO         = "#708a9a"

# Text — warm cream hierarchy
TEXT_TITLE   = "#e8e4dc"    # primary / headings
TEXT_PRIMARY = "#e8e4dc"    # body text
TEXT_SEC     = "#a09890"    # secondary / muted labels
TEXT_HINT    = "#6a6460"    # disabled / placeholder

# Hairline border for use directly in QSS strings (not in lerp)
HAIR = "rgba(232,228,220,0.08)"


# ─── Color Utilities ──────────────────────────────────────────────────────────
def lerp(c1: str, c2: str, t: float) -> str:
    """Linear interpolation between two hex color strings."""
    def _p(c: str):
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r1, g1, b1 = _p(c1)
    r2, g2, b2 = _p(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t), int(b1 + (b2 - b1) * t))


def darken(c: str, amount: float = 0.15) -> str:
    return lerp(c, "#000000", amount)


def lighten(c: str, amount: float = 0.12) -> str:
    return lerp(c, "#ffffff", amount)


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
def fade_in(widget: QWidget, duration: int = 260) -> QPropertyAnimation:
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
GLOBAL_QSS = f"""
QWidget {{
    background-color: {BG_DEEP};
    color: {TEXT_PRIMARY};
    font-family: "Courier New", monospace;
    font-size: 13px;
}}
QMainWindow {{ background-color: {BG_DEEP}; }}
QLineEdit {{
    background-color: {BG_FLOAT};
    border: 1px solid {HAIR};
    border-radius: 0px;
    padding: 8px 12px;
    color: {TEXT_PRIMARY};
    font-family: "Courier New", monospace;
    font-size: 13px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
    background-color: {BG_FLOAT};
}}
QProgressBar {{
    background-color: {HAIR};
    border: none;
    border-radius: 0px;
    text-align: center;
    color: transparent;
    height: 4px;
}}
QProgressBar::chunk {{
    background-color: {SUCCESS};
    border-radius: 0px;
}}
QListWidget {{
    background-color: {BG_SURFACE};
    border: 1px solid {HAIR};
    border-radius: 0px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-radius: 0px;
    color: {TEXT_PRIMARY};
    font-size: 12px;
    font-family: "Courier New", monospace;
}}
QListWidget::item:selected {{
    background-color: rgba(74,124,89,0.18);
    color: {TEXT_TITLE};
}}
QListWidget::item:hover:!selected {{ background-color: rgba(232,228,220,0.04); }}
QScrollBar:vertical {{ background: transparent; width: 4px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BG_FLOAT};
    border-radius: 2px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 4px; }}
QScrollBar::handle:horizontal {{
    background: {BG_FLOAT};
    border-radius: 2px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QRadioButton {{
    color: {TEXT_PRIMARY};
    spacing: 8px;
    font-size: 12px;
    font-family: "Courier New", monospace;
}}
QRadioButton::indicator {{
    width: 14px; height: 14px; border-radius: 7px;
    border: 1px solid {HAIR}; background-color: {BG_ELEVATED};
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 8px;
    font-size: 12px;
    font-family: "Courier New", monospace;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 0px;
    border: 1px solid {HAIR}; background-color: {BG_ELEVATED};
}}
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
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {HAIR};
    border-radius: 0px;
    padding: 4px 8px;
    font-family: "Courier New", monospace;
    font-size: 11px;
}}
QMessageBox {{
    background-color: {BG_SURFACE};
}}
"""


# ─── Components ───────────────────────────────────────────────────────────────

class GlassCard(QFrame):
    """Flat dark card with hairline border. No drop-shadow by default."""

    def __init__(self, parent=None, radius=0, bg=BG_SURFACE,
                 border_color=None, shadow=False):
        super().__init__(parent)
        self._bg = bg
        self._border = border_color or HAIR
        self._radius = radius
        self._apply_style()
        if shadow:
            eff = QGraphicsDropShadowEffect(self)
            eff.setBlurRadius(16)
            eff.setOffset(0.0, 2.0)
            eff.setColor(QColor(0, 0, 0, 40))
            self.setGraphicsEffect(eff)

    def _apply_style(self):
        self.setStyleSheet(
            f"QFrame {{ background-color: {self._bg}; "
            f"border: 1px solid {self._border}; border-radius: {self._radius}px; }}"
            f"QLabel {{ background: transparent; border: none; }}"
            f"QProgressBar {{ border: none; }}"
        )

    def set_border_color(self, color: str):
        self._border = color
        self._apply_style()


class PillButton(QPushButton):
    """Solid pill button — kept for back/nav buttons and primary actions."""

    def __init__(self, text="", color=ACCENT, hover_color=None,
                 text_color=TEXT_TITLE, parent=None):
        super().__init__(text, parent)
        self._base  = color
        self._hover = hover_color or lighten(color, 0.10)
        self._txt   = text_color
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._paint(self._base)

    def _paint(self, bg: str):
        self.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; color: {self._txt}; "
            f"border: 1px solid {HAIR}; border-radius: 0px; padding: 8px 20px; "
            f"font-family: 'Courier New', monospace; font-size: 11px; "
            f"letter-spacing: 1px; text-transform: uppercase; }}"
            f"QPushButton:disabled {{ background-color: {BG_BORDER}; "
            f"color: {TEXT_HINT}; }}")

    def enterEvent(self, e):
        self._paint(self._hover); super().enterEvent(e)

    def leaveEvent(self, e):
        self._paint(self._base); super().leaveEvent(e)


class OutlineButton(QPushButton):
    """Flat outline button — no fill, 1px cream border, uppercase monospace."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._paint_normal()

    def _paint_normal(self):
        self.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {TEXT_PRIMARY}; "
            f"border: 1px solid rgba(232,228,220,0.25); border-radius: 0px; "
            f"padding: 7px 16px; "
            f"font-family: 'Courier New', monospace; font-size: 10px; "
            f"letter-spacing: 1px; }}"
            f"QPushButton:disabled {{ color: {TEXT_HINT}; "
            f"border-color: rgba(232,228,220,0.08); }}")

    def _paint_hover(self):
        self.setStyleSheet(
            f"QPushButton {{ background-color: rgba(232,228,220,0.06); "
            f"color: {TEXT_TITLE}; "
            f"border: 1px solid rgba(232,228,220,0.35); border-radius: 0px; "
            f"padding: 7px 16px; "
            f"font-family: 'Courier New', monospace; font-size: 10px; "
            f"letter-spacing: 1px; }}"
            f"QPushButton:disabled {{ color: {TEXT_HINT}; }}")

    def enterEvent(self, e):
        self._paint_hover(); super().enterEvent(e)

    def leaveEvent(self, e):
        self._paint_normal(); super().leaveEvent(e)


class DangerButton(OutlineButton):
    """Outline button with red tint."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._paint_danger_normal()

    def _paint_danger_normal(self):
        self.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {DANGER}; "
            f"border: 1px solid rgba(138,64,64,0.4); border-radius: 0px; "
            f"padding: 7px 16px; "
            f"font-family: 'Courier New', monospace; font-size: 10px; "
            f"letter-spacing: 1px; }}"
            f"QPushButton:disabled {{ color: {TEXT_HINT}; }}")

    def _paint_danger_hover(self):
        self.setStyleSheet(
            f"QPushButton {{ background-color: rgba(138,64,64,0.12); "
            f"color: #b05050; "
            f"border: 1px solid rgba(138,64,64,0.6); border-radius: 0px; "
            f"padding: 7px 16px; "
            f"font-family: 'Courier New', monospace; font-size: 10px; "
            f"letter-spacing: 1px; }}")

    def enterEvent(self, e):
        self._paint_danger_hover(); super(OutlineButton, self).enterEvent(e)

    def leaveEvent(self, e):
        self._paint_danger_normal(); super(OutlineButton, self).leaveEvent(e)


class PulsingDot(QLabel):
    """Small dot that pulses between active green and dim green every 1000 ms."""

    def __init__(self, on_color=SUCCESS, off_color="#2a4a35", size=7, parent=None):
        super().__init__(parent)
        self._on    = on_color
        self._off   = off_color
        self._size  = size
        self._state = True
        r = (size + 4) // 2
        self.setFixedSize(size + 4, size + 4)
        self._r = r
        self._update()
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._toggle)
        self._timer.start()

    def _toggle(self):
        self._state = not self._state
        self._update()

    def _update(self):
        c = self._on if self._state else self._off
        self.setStyleSheet(
            f"QLabel {{ background-color: {c}; border-radius: {self._r}px; border: none; }}")


class CardButton(QFrame):
    """Large animated tile for the home screen."""

    clicked = pyqtSignal()

    def __init__(self, icon="", title="", subtitle="", accent=ACCENT, parent=None):
        super().__init__(parent)
        self._accent = accent
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumSize(200, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(8)
        lay.setContentsMargins(28, 24, 28, 24)

        self._icon_lbl = QLabel(icon)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet(
            f"font-size: 32px; color: {accent}; background: transparent; border: none;")

        self._title_lbl = QLabel(title)
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_lbl.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {TEXT_TITLE};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            f" background: transparent; border: none;")

        self._sub_lbl = QLabel(subtitle)
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_lbl.setWordWrap(True)
        self._sub_lbl.setStyleSheet(
            f"font-size: 11px; color: {TEXT_SEC};"
            f" font-family: 'Courier New', monospace;"
            f" background: transparent; border: none;")

        lay.addWidget(self._icon_lbl)
        lay.addWidget(self._title_lbl)
        lay.addWidget(self._sub_lbl)

        self._set_normal()

    def _set_normal(self):
        self.setStyleSheet(
            f"QFrame {{ background-color: {BG_SURFACE};"
            f" border: 1px solid {HAIR}; border-radius: 0px; }}"
            f"QLabel {{ background: transparent; border: none; }}")

    def _set_hovered(self):
        self.setStyleSheet(
            f"QFrame {{ background-color: {BG_ELEVATED};"
            f" border: 1px solid rgba(232,228,220,0.15); border-radius: 0px; }}"
            f"QLabel {{ background: transparent; border: none; }}")

    def _set_pressed(self):
        self.setStyleSheet(
            f"QFrame {{ background-color: {BG_ACTIVE};"
            f" border: 1px solid rgba(74,124,89,0.5); border-radius: 0px; }}"
            f"QLabel {{ background: transparent; border: none; }}")

    def enterEvent(self, e):   self._set_hovered(); super().enterEvent(e)
    def leaveEvent(self, e):   self._set_normal();  super().leaveEvent(e)
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self._set_pressed()
        super().mousePressEvent(e)
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._set_hovered(); self.clicked.emit()
        super().mouseReleaseEvent(e)


class CameraFrame(QLabel):
    """QLabel that clips camera feed to rounded corners and draws LIVE overlay."""

    def __init__(self, w=640, h=480, radius=0, parent=None):
        super().__init__(parent)
        self._radius = radius
        self._pix: QPixmap | None = None
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background-color: {BG_SURFACE};")

    def set_frame(self, pix: QPixmap | None):
        self._pix = pix
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._radius > 0:
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), self._radius, self._radius)
            p.setClipPath(path)

        if self._pix:
            scaled = self._pix.scaled(self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)

            # LIVE · 30fps overlay — top-left corner
            p.setPen(QColor(232, 228, 220, 200))
            font = QFont("Courier New", 9)
            p.setFont(font)
            p.drawText(10, 18, "LIVE \u00b7 30fps")
        else:
            p.fillRect(self.rect(), QColor(BG_SURFACE))
            p.setPen(QColor(TEXT_HINT))
            p.setFont(QFont("Courier New", 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Camera Preview\nFeed activates when this screen opens")
        p.end()


class StatusDot(QLabel):
    """Tiny colored dot status indicator."""

    def __init__(self, color=TEXT_HINT, size=7, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size + 4, size + 4)
        self.set_color(color)

    def set_color(self, color: str):
        r = (self._size + 4) // 2
        self.setStyleSheet(
            f"QLabel {{ background-color: {color}; border-radius: {r}px; border: none; }}")


class SectionLabel(QLabel):
    """Small ALL-CAPS monospace section header."""

    def __init__(self, text="", parent=None):
        super().__init__(text.upper(), parent)
        self.setStyleSheet(
            f"QLabel {{ color: {TEXT_HINT}; font-size: 9px; font-weight: 400;"
            f" font-family: 'Courier New', monospace;"
            f" letter-spacing: 2px; background: transparent; border: none; }}")


class HSep(QFrame):
    """Thin horizontal separator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background-color: {BG_BORDER}; border: none;")
