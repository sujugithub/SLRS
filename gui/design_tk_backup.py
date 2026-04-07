"""
Premium design system — Apple HIG-inspired dark space theme.
Import this in every screen for consistent styling.
"""
import tkinter as tk
import platform

# ─── Color Tokens ─────────────────────────────────────────────────────────────
BG_DEEP     = "#07070E"   # Window root — deepest black
BG_BASE     = "#0C0C1A"   # Base surface
BG_SURFACE  = "#111127"   # Card/panel bg
BG_ELEVATED = "#181833"   # Elevated card (glass sim)
BG_FLOAT    = "#1E1E3E"   # Floating / hover layer
BG_BORDER   = "#252548"   # Glass border
BG_ACTIVE   = "#2E2E5C"   # Active/focus border

ACCENT       = "#5E5CE6"   # Primary action (Apple Indigo)
ACCENT_HOVER = "#7977FF"   # Hover
ACCENT_DARK  = "#4442C0"   # Pressed
ACCENT_TINT  = "#12123A"   # Subtle bg tint

SUCCESS      = "#30D158"   # Apple system green
SUCCESS_DIM  = "#1C6B31"
SUCCESS_TINT = "#0A1C0F"
WARNING      = "#FF9F0A"   # Apple amber
DANGER       = "#FF453A"   # Apple red
DANGER_DIM   = "#8A2222"
DANGER_TINT  = "#2A0A0A"
INFO         = "#64D2FF"   # Apple blue

TEXT_TITLE   = "#F5F5F7"   # Near-white (Apple's actual value)
TEXT_PRIMARY = "#E5E5EA"
TEXT_SEC     = "#8E8EA0"
TEXT_HINT    = "#464660"


# ─── Font System ──────────────────────────────────────────────────────────────
def _f(size, weight="normal"):
    """Best available font for the platform."""
    on_mac = platform.system() == "Darwin"
    fam = ("SF Pro Display" if size >= 20 else "SF Pro Text") if on_mac else "Helvetica Neue"
    if weight in ("bold", "italic"):
        return (fam, size, weight)
    return (fam, size)


F_DISPLAY = _f(30, "bold")
F_TITLE   = _f(20, "bold")
F_HEAD    = _f(15, "bold")
F_BODY_LB = _f(14, "bold")
F_BODY_L  = _f(14)
F_BODY_B  = _f(13, "bold")
F_BODY    = _f(13)
F_CAP_B   = _f(11, "bold")
F_CAP     = _f(11)
F_SMALL   = _f(10)
F_SIGN    = _f(38, "bold")   # Large sign name display


# ─── Color Utilities ──────────────────────────────────────────────────────────
def _rgb(c):
    h = c.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(r, g, b):
    return (f"#{int(max(0, min(255, r))):02x}"
            f"{int(max(0, min(255, g))):02x}"
            f"{int(max(0, min(255, b))):02x}")


def lerp(c1, c2, t):
    r1, g1, b1 = _rgb(c1)
    r2, g2, b2 = _rgb(c2)
    return _hex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def darken(c, f=0.75):
    r, g, b = _rgb(c)
    return _hex(r * f, g * f, b * f)


# ─── Canvas Drawing ───────────────────────────────────────────────────────────
def rrect(canvas, x1, y1, x2, y2, r=12, fill=BG_ELEVATED):
    """Draw a filled rounded rectangle on a canvas."""
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    kw = dict(fill=fill, outline=fill, style="pieslice")
    canvas.create_arc(x1,      y1,      x1 + 2*r, y1 + 2*r, start=90,  extent=90, **kw)
    canvas.create_arc(x2 - 2*r, y1,    x2,        y1 + 2*r, start=0,   extent=90, **kw)
    canvas.create_arc(x1,      y2 - 2*r, x1 + 2*r, y2,      start=180, extent=90, **kw)
    canvas.create_arc(x2 - 2*r, y2 - 2*r, x2,    y2,        start=270, extent=90, **kw)
    canvas.create_rectangle(x1 + r, y1,   x2 - r, y2,   fill=fill, outline="")
    canvas.create_rectangle(x1,     y1 + r, x2,   y2 - r, fill=fill, outline="")


def rrect_border(canvas, x1, y1, x2, y2, r=12, color=BG_BORDER, width=1):
    """Draw the border outline of a rounded rectangle via smooth polygon."""
    canvas.create_polygon(
        x1 + r, y1,   x2 - r, y1,
        x2,     y1 + r, x2,   y2 - r,
        x2 - r, y2,   x1 + r, y2,
        x1,     y2 - r, x1,   y1 + r,
        smooth=True, fill="", outline=color, width=width,
    )


# ─── Composite Widgets ────────────────────────────────────────────────────────
class GlassCard(tk.Frame):
    """
    Premium glass card: dark tinted surface with a subtle 1px border.
    Add child widgets to `card.body`.
    """
    def __init__(self, parent, bg=BG_ELEVATED, border_color=BG_BORDER, **kwargs):
        super().__init__(parent, bg=border_color, **kwargs)
        self.body = tk.Frame(self, bg=bg)
        self.body.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)


class PillButton(tk.Canvas):
    """
    Animated pill-shaped button with smooth hover color transitions and
    a subtle drop shadow + glass sheen. Resizes with parent (fill=X safe).
    """
    def __init__(self, parent, text, command,
                 color=ACCENT, hover_color=None, text_color=TEXT_TITLE,
                 height=42, btn_width=None, icon="", btn_font=None, **kwargs):
        h = height
        w = btn_width or 180
        parent_bg = parent.cget("bg")
        super().__init__(parent, width=w, height=h + 4, bg=parent_bg,
                         highlightthickness=0, **kwargs)
        self._text    = text
        self._icon    = icon
        self._cmd     = command
        self._c       = color
        self._ch      = hover_color or lerp(color, "#ffffff", 0.18)
        self._tc      = text_color
        self._h       = h
        self._font    = btn_font or F_BODY_B
        self._cur     = color
        self._job     = None
        self._enabled = True

        self._paint(color)
        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<Button-1>",        self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>",       self._on_resize)
        self.config(cursor="hand2")

    def _paint(self, color):
        self.delete("all")
        w = self.winfo_width()
        if w < 4:
            w = int(self["width"]) if self["width"] else 180
        h = self._h
        r = h // 2

        # Drop shadow
        for i in range(3, 0, -1):
            sv = lerp(darken(color, 0.25), self.cget("bg"), 0.4)
            self.create_rectangle(i, i + 3, w - i, h + i + 1, fill=sv, outline="")

        # Pill body
        a = dict(fill=color, outline=color, style="pieslice")
        self.create_arc(0,       0, 2 * r, h, start=90,  extent=180, **a)
        self.create_arc(w - 2*r, 0, w,     h, start=270, extent=180, **a)
        self.create_rectangle(r, 0, w - r, h, fill=color, outline="")

        # Top glass sheen
        sheen = lerp(color, "#ffffff", 0.08)
        sh = max(2, h // 4)
        self.create_rectangle(r, 1, w - r, sh, fill=sheen, outline="")

        # Label
        label = f"{self._icon}  {self._text}" if self._icon else self._text
        self.create_text(w // 2, h // 2, text=label,
                         fill=self._tc, font=self._font, anchor="center")

    def _anim(self, target, steps=7):
        if self._job:
            self.after_cancel(self._job)
        start = self._cur

        def _step(i):
            c = lerp(start, target, i / steps)
            self._cur = c
            self._paint(c)
            if i < steps:
                self._job = self.after(12, lambda: _step(i + 1))

        _step(1)

    def _on_enter(self, e):
        if self._enabled:
            self._anim(self._ch)

    def _on_leave(self, e):
        if self._enabled:
            self._anim(self._c)

    def _on_press(self, e):
        if self._enabled:
            self._paint(darken(self._c, 0.82))

    def _on_release(self, e):
        if not self._enabled:
            return
        w = self.winfo_width() or int(self["width"])
        if 0 <= e.x <= w and 0 <= e.y <= self._h:
            self._paint(self._ch)
            if self._cmd:
                self._cmd()
        else:
            self._paint(self._c)
            self._cur = self._c

    def _on_resize(self, e):
        if e.width > 4:
            self._paint(self._cur)

    def enable(self):
        self._enabled = True
        self._paint(self._c)
        self.config(cursor="hand2")
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def disable(self):
        self._enabled = False
        self._paint(BG_FLOAT)
        self._cur = BG_FLOAT
        self.config(cursor="arrow")
        self.unbind("<Enter>")
        self.unbind("<Leave>")

    def set_text(self, text):
        self._text = text
        self._paint(self._cur)


class StatusDot(tk.Canvas):
    """A tiny filled circle for status indication."""
    def __init__(self, parent, color=TEXT_HINT, size=8, **kwargs):
        bg = parent.cget("bg")
        super().__init__(parent, width=size + 2, height=size + 2,
                         bg=bg, highlightthickness=0, **kwargs)
        self._size = size
        self.set_color(color)

    def set_color(self, color):
        self.delete("all")
        s = self._size
        self.create_oval(1, 1, s + 1, s + 1, fill=color, outline="")
