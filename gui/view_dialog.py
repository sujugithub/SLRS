"""
View dialog for SignBridge.

Lists every sign currently stored under ``data/custom/`` along with sample
count and last training accuracy. Each row exposes "Add samples" and
"Delete" actions.

Sample-count colour coding:
    < 20  → DANGER  (undertrained)
    20-60 → SUCCESS (healthy)
    > 60  → WARNING (overfit risk)
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QScrollArea,
    QVBoxLayout, QWidget,
)

import numpy as np

from config import CUSTOM_DATA_DIR
from core import training_meta
from gui.design import (
    ACCENT, BG_BASE, BG_DEEP, BG_ELEVATED, BG_SURFACE, HAIR,
    SUCCESS, WARNING, DANGER, TEXT_HINT, TEXT_PRIMARY, TEXT_SEC, TEXT_TITLE,
    FONT_MONO, FONT_DISPLAY,
    GlassCard, HSep, OutlineButton, DangerButton, SectionLabel,
)


def _chip(text: str, color: str) -> QLabel:
    """Small inline chip with a colored hairline border."""
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"QLabel {{ color: {color};"
        f" font-size: 10px;"
        f" font-family: '{FONT_MONO}', monospace;"
        f" letter-spacing: 1px;"
        f" border: 1px solid {color};"
        f" padding: 2px 8px;"
        f" background: transparent; }}"
    )
    return lbl


class ViewDialog(QWidget):
    """Page rendering the sign list."""

    add_samples_requested = pyqtSignal(str)
    delete_requested      = pyqtSignal(str)
    back_requested        = pyqtSignal()

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
        nav_lay.setContentsMargins(28, 0, 28, 0)
        nav_lay.setSpacing(12)

        nav_lay.addStretch(1)

        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_HINT};"
            f" font-family: '{FONT_MONO}', monospace;"
            f" letter-spacing: 2px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        nav_lay.addWidget(self._count_lbl, 0, Qt.AlignmentFlag.AlignRight)
        root.addWidget(nav)

        # Body
        body = QWidget()
        body.setStyleSheet(f"background-color: {BG_DEEP};")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(40, 28, 40, 28)
        body_lay.setSpacing(0)
        root.addWidget(body, 1)

        # Scroll area + inner column
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("border: none; background: transparent;")

        self._inner = QWidget()
        self._inner.setStyleSheet("background: transparent;")
        self._inner_lay = QVBoxLayout(self._inner)
        self._inner_lay.setContentsMargins(0, 0, 0, 0)
        self._inner_lay.setSpacing(8)
        self._inner_lay.addStretch(1)
        self._scroll.setWidget(self._inner)
        body_lay.addWidget(self._scroll)

    # ─── Public API ───────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Re-scan ``data/custom/`` and rebuild the list."""
        # Clear existing rows
        while self._inner_lay.count() > 1:
            item = self._inner_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        signs = self._scan_signs()
        meta = training_meta.load_meta() or {}
        per_sign = meta.get("per_sign", {}) if isinstance(meta, dict) else {}

        self._count_lbl.setText(f"{len(signs)} signs")

        if not signs:
            empty = QLabel(
                "No custom signs yet.\n\n"
                "Use ☰ → Train to record your first sign."
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(
                f"font-size: 12px; color: {TEXT_HINT};"
                f" font-family: '{FONT_MONO}', monospace; line-height: 1.6;"
                " background: transparent; border: none; padding: 60px 0;"
            )
            self._inner_lay.insertWidget(0, empty)
            return

        for sign_name, n_samples in signs:
            entry = per_sign.get(sign_name, {}) if isinstance(per_sign, dict) else {}
            acc = entry.get("accuracy")
            row = self._build_row(sign_name, n_samples, acc)
            self._inner_lay.insertWidget(self._inner_lay.count() - 1, row)

    # ─── Internals ────────────────────────────────────────────────────────────
    def _scan_signs(self) -> list[tuple[str, int]]:
        """Return [(sign_name, num_samples), ...] sorted alphabetically."""
        results: list[tuple[str, int]] = []
        if not os.path.isdir(CUSTOM_DATA_DIR):
            return results
        try:
            entries = sorted(os.listdir(CUSTOM_DATA_DIR))
        except OSError:
            return results
        for name in entries:
            sign_dir = os.path.join(CUSTOM_DATA_DIR, name)
            if not os.path.isdir(sign_dir):
                continue
            npy_path = os.path.join(sign_dir, "features.npy")
            if not os.path.isfile(npy_path):
                continue
            try:
                arr = np.load(npy_path, mmap_mode="r")
                n = int(arr.shape[0])
            except Exception:
                n = 0
            results.append((name, n))
        return results

    def _build_row(self, sign: str, n: int, acc) -> QFrame:
        card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        lay = QHBoxLayout(card)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(12)

        # Sign name
        name_lbl = QLabel(sign.replace("_", " ").upper())
        name_lbl.setStyleSheet(
            f"font-size: 16px; color: {TEXT_TITLE};"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            " background: transparent; border: none;"
        )
        name_lbl.setMinimumWidth(180)
        lay.addWidget(name_lbl)

        # Sample count chip
        if n < 20:
            chip_color = DANGER
            chip_text = f"{n} samples · undertrained"
        elif n <= 60:
            chip_color = SUCCESS
            chip_text = f"{n} samples · healthy"
        else:
            chip_color = WARNING
            chip_text = f"{n} samples · overfit risk"
        lay.addWidget(_chip(chip_text, chip_color))

        # Accuracy chip
        if isinstance(acc, (int, float)):
            acc_pct = float(acc) * 100
            if acc_pct >= 85:
                acc_color = SUCCESS
            elif acc_pct >= 65:
                acc_color = WARNING
            else:
                acc_color = DANGER
            lay.addWidget(_chip(f"acc: {acc_pct:.0f}%", acc_color))
        else:
            lay.addWidget(_chip("acc: —", TEXT_HINT))

        lay.addStretch(1)

        # Add samples button
        add_btn = OutlineButton("+ ADD SAMPLES")
        add_btn.clicked.connect(lambda _checked=False, s=sign:
                                self.add_samples_requested.emit(s))
        lay.addWidget(add_btn)

        # Delete button
        del_btn = DangerButton("DELETE")
        del_btn.clicked.connect(lambda _checked=False, s=sign:
                                self._on_delete_clicked(s))
        lay.addWidget(del_btn)

        return card

    def _on_delete_clicked(self, sign_name: str) -> None:
        reply = QMessageBox.question(
            self, "Delete Sign",
            f"Permanently delete \"{sign_name.replace('_', ' ').upper()}\""
            f" and all its training samples?\n\n"
            f"The model will be retrained automatically.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(sign_name)
