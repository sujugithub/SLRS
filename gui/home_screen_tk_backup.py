"""
Home screen — premium Apple HIG-inspired redesign.
Features a gradient hero canvas, glassmorphism feature cards,
and a clean signs list panel.
"""
import os
import shutil
import tkinter as tk
from tkinter import messagebox, ttk

from config import PRETRAINED_DATA_DIR, CUSTOM_DATA_DIR, DEFAULT_MODEL_FILE
from gui.design import (
    BG_DEEP, BG_BASE, BG_ELEVATED, BG_SURFACE, BG_BORDER, BG_FLOAT,
    ACCENT, ACCENT_TINT, SUCCESS, DANGER, TEXT_TITLE, TEXT_PRIMARY, TEXT_SEC, TEXT_HINT,
    F_DISPLAY, F_HEAD, F_BODY_B, F_BODY, F_CAP, F_SMALL, _f,
    lerp, GlassCard, PillButton, StatusDot,
)


class HomeScreen:
    """Premium home screen with hero section and feature navigation cards."""

    def __init__(self, parent, on_train_click=None, on_predict_click=None):
        self.parent = parent
        self.on_train_click = on_train_click
        self.on_predict_click = on_predict_click
        self.frame = tk.Frame(parent, bg=BG_DEEP)
        self._build_ui()

    # ─── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # Hero canvas — gradient + logo + title
        self._hero = tk.Canvas(self.frame, height=182, bg=BG_DEEP,
                               highlightthickness=0)
        self._hero.pack(fill=tk.X)
        self._hero.bind("<Configure>",
                        lambda e: self._draw_hero(e.width, 182))
        self.frame.after(60, lambda: self._draw_hero(
            self._hero.winfo_width() or 900, 182))

        # Feature cards row (Train / Detect)
        cards = tk.Frame(self.frame, bg=BG_DEEP)
        cards.pack(fill=tk.X, padx=28, pady=(4, 16))
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)

        self._build_feature_card(
            cards, col=0,
            icon="✦", title="Train New Sign",
            desc="Capture hand gestures and build\nyour personal AI vocabulary",
            btn_text="Get Started", btn_color=ACCENT,
            callback=self._on_train,
        )
        self._build_feature_card(
            cards, col=1,
            icon="◎", title="Live Detection",
            desc="Recognize signs in real time\nwith sentence composition",
            btn_text="Start Detection", btn_color=SUCCESS,
            btn_hover=lerp(SUCCESS, "#ffffff", 0.16),
            callback=self._on_predict,
        )

        # Section header: Available Signs
        hdr = tk.Frame(self.frame, bg=BG_DEEP)
        hdr.pack(fill=tk.X, padx=28, pady=(0, 6))

        tk.Label(hdr, text="Available Signs",
                 font=F_HEAD, fg=TEXT_TITLE, bg=BG_DEEP).pack(side=tk.LEFT)
        self._count_lbl = tk.Label(hdr, text="",
                                   font=F_SMALL, fg=TEXT_SEC, bg=BG_DEEP)
        self._count_lbl.pack(side=tk.LEFT, padx=(8, 0))

        PillButton(
            hdr, text="✕ Delete Selected", command=self._delete_selected_sign,
            color=DANGER, height=28, btn_width=140, btn_font=F_SMALL,
        ).pack(side=tk.RIGHT)

        # Signs panel — glass card with listbox
        card = GlassCard(self.frame, bg=BG_SURFACE, border_color=BG_BORDER)
        card.pack(fill=tk.BOTH, expand=True, padx=28, pady=(0, 6))
        inner = card.body

        self.sign_listbox = tk.Listbox(
            inner,
            font=F_BODY,
            bg=BG_SURFACE,
            fg=TEXT_PRIMARY,
            selectbackground=ACCENT,
            selectforeground=TEXT_TITLE,
            borderwidth=0,
            highlightthickness=0,
            activestyle="none",
        )
        self.sign_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                               padx=8, pady=8)

        sb = ttk.Scrollbar(inner, orient=tk.VERTICAL,
                           command=self.sign_listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=8, padx=(0, 4))
        self.sign_listbox.config(yscrollcommand=sb.set)

        # Status bar with dot indicator
        status_row = tk.Frame(self.frame, bg=BG_DEEP)
        status_row.pack(fill=tk.X, padx=28, pady=(0, 16))

        self._dot = StatusDot(status_row, color=TEXT_HINT, size=7)
        self._dot.pack(side=tk.LEFT, pady=1)

        self.status_label = tk.Label(status_row, text="",
                                     font=F_SMALL, fg=TEXT_SEC, bg=BG_DEEP)
        self.status_label.pack(side=tk.LEFT, padx=(6, 0))

        self.refresh_sign_list()

    def _draw_hero(self, w, h):
        """Render gradient background + app identity on the hero canvas."""
        if w < 2:
            return
        c = self._hero
        c.delete("all")

        # Multi-stop gradient: deep black → base → subtle accent tint
        steps = 52
        for i in range(steps):
            t = i / steps
            if t < 0.55:
                col = lerp(BG_DEEP, BG_BASE, t / 0.55)
            else:
                col = lerp(BG_BASE, lerp(BG_BASE, ACCENT_TINT, 0.45),
                           (t - 0.55) / 0.45)
            y1 = int(i * h / steps)
            y2 = int((i + 1) * h / steps) + 1
            c.create_rectangle(0, y1, w, y2, fill=col, outline="")

        cx = w // 2

        # Diamond glyph — subtle accent focal point
        c.create_text(cx, 38, text="◆", font=_f(18), fill=ACCENT, anchor="center")

        # App name — large display type
        c.create_text(cx, 80, text="SignAI",
                      font=F_DISPLAY, fill=TEXT_TITLE, anchor="center")

        # Tagline
        c.create_text(cx, 114,
                      text="Sign Language Recognition  ·  Powered by AI",
                      font=F_BODY, fill=TEXT_SEC, anchor="center")

        # Thin separator line
        sep = lerp(BG_BASE, BG_BORDER, 0.55)
        c.create_line(cx - 90, 144, cx + 90, 144, fill=sep, width=1)

    def _build_feature_card(self, parent, col, icon, title, desc,
                             btn_text, callback, btn_color=ACCENT,
                             btn_hover=None):
        """Build a glassmorphism feature card in a grid column."""
        card = GlassCard(parent, bg=BG_ELEVATED, border_color=BG_BORDER)
        card.grid(row=0, column=col, sticky="nsew",
                  padx=(0, 8) if col == 0 else (8, 0))

        body = tk.Frame(card.body, bg=BG_ELEVATED)
        body.pack(fill=tk.BOTH, expand=True, padx=22, pady=20)

        # Icon glyph
        tk.Label(body, text=icon, font=_f(22), fg=ACCENT,
                 bg=BG_ELEVATED).pack(anchor=tk.W)
        tk.Frame(body, height=10, bg=BG_ELEVATED).pack()

        # Card title
        tk.Label(body, text=title, font=F_HEAD,
                 fg=TEXT_TITLE, bg=BG_ELEVATED, anchor=tk.W).pack(fill=tk.X)
        tk.Frame(body, height=5, bg=BG_ELEVATED).pack()

        # Card description
        tk.Label(body, text=desc, font=F_CAP, fg=TEXT_SEC,
                 bg=BG_ELEVATED, anchor=tk.W,
                 justify=tk.LEFT).pack(fill=tk.X)
        tk.Frame(body, height=18, bg=BG_ELEVATED).pack()

        # CTA button — pill shape, fills card width
        PillButton(body, text=btn_text, command=callback,
                   color=btn_color, hover_color=btn_hover,
                   height=38, btn_width=1,
                   btn_font=F_BODY_B).pack(fill=tk.X)

    # ─── Public API ───────────────────────────────────────────────────────────
    def show(self):
        self.refresh_sign_list()
        self.frame.pack(fill=tk.BOTH, expand=True)

    def hide(self):
        self.frame.pack_forget()

    def refresh_sign_list(self):
        self.sign_listbox.delete(0, tk.END)
        signs = set()

        for data_dir in (PRETRAINED_DATA_DIR, CUSTOM_DATA_DIR):
            if not os.path.isdir(data_dir):
                continue
            for entry in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, entry)):
                    signs.add(entry)

        for sign in sorted(signs):
            self.sign_listbox.insert(tk.END, f"   {sign}")

        count = len(signs)
        if count:
            self._dot.set_color(SUCCESS)
            self._count_lbl.config(text=f"({count})")
            noun = "sign" if count == 1 else "signs"
            self.status_label.config(
                text=f"{count} {noun} ready for detection")
        else:
            self._dot.set_color(TEXT_HINT)
            self._count_lbl.config(text="")
            self.sign_listbox.insert(
                tk.END, "   No signs found — train one to get started")
            self.status_label.config(
                text="Train your first sign to get started")

    # ─── Callbacks ────────────────────────────────────────────────────────────
    def _delete_selected_sign(self):
        """Delete all training data for the selected sign and invalidate the model."""
        sel = self.sign_listbox.curselection()
        if not sel:
            messagebox.showinfo("No Selection",
                                "Select a sign from the list first.")
            return

        raw = self.sign_listbox.get(sel[0]).strip()
        if not raw or raw.startswith("No signs"):
            return

        confirmed = messagebox.askyesno(
            "Delete Sign Data",
            f"Delete all training data for  \u201c{raw}\u201d?\n\n"
            "This cannot be undone. You will need to retrain the model "
            "afterwards.",
            icon="warning",
        )
        if not confirmed:
            return

        deleted = False
        for data_dir in (CUSTOM_DATA_DIR, PRETRAINED_DATA_DIR):
            sign_dir = os.path.join(data_dir, raw)
            if os.path.isdir(sign_dir):
                shutil.rmtree(sign_dir)
                deleted = True

        if deleted:
            # Remove the trained model so it is rebuilt on the next train run
            if os.path.isfile(DEFAULT_MODEL_FILE):
                os.remove(DEFAULT_MODEL_FILE)
            self.refresh_sign_list()
            self.status_label.config(
                text=f"\u2713 \u201c{raw}\u201d deleted — re-train to update the model",
                fg=DANGER,
            )
        else:
            messagebox.showerror("Error", f"Could not find data for \u201c{raw}\u201d.")

    def _on_train(self):
        if self.on_train_click:
            self.on_train_click()

    def _on_predict(self):
        if self.on_predict_click:
            self.on_predict_click()
