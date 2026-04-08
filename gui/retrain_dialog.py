"""
Retrain dialog for SignBridge.

A page with a single "RETRAIN ALL" button that re-runs Random Forest
training over every sign currently under ``data/custom/`` and writes
``data/training_meta.json`` with overall + per-sign accuracy.

The actual retraining runs on a worker thread (RetrainWorker, defined in
main.py); this dialog only renders the trigger, the indeterminate progress
bar, and the results panel.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QScrollArea, QVBoxLayout, QWidget,
)

from gui.design import (
    ACCENT, BG_BASE, BG_DEEP, BG_FLOAT, BG_SURFACE, HAIR,
    SUCCESS, WARNING, DANGER, TEXT_HINT, TEXT_PRIMARY, TEXT_SEC, TEXT_TITLE,
    FONT_MONO, FONT_DISPLAY,
    GlassCard, HSep, PillButton, OutlineButton, SectionLabel,
)


class RetrainDialog(QWidget):
    """Page rendering the Retrain controls + results panel."""

    retrain_requested = pyqtSignal()
    back_requested    = pyqtSignal()

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
        root.addWidget(nav)

        # Body
        body = QWidget()
        body.setStyleSheet(f"background-color: {BG_DEEP};")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(40, 28, 40, 28)
        body_lay.setSpacing(16)
        root.addWidget(body, 1)

        # Header card with the action button
        action_card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        ac_lay = QVBoxLayout(action_card)
        ac_lay.setContentsMargins(28, 24, 28, 24)
        ac_lay.setSpacing(12)

        explainer = QLabel(
            "Re-trains the Random Forest model on every sign currently "
            "stored under data/custom/. Per-sign accuracy is captured and "
            "shown in the View screen after retraining."
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet(
            f"font-size: 11px; color: {TEXT_SEC};"
            f" font-family: '{FONT_MONO}', monospace;"
            " background: transparent; border: none;"
        )
        ac_lay.addWidget(explainer)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._retrain_btn = PillButton("RETRAIN ALL", color=ACCENT)
        self._retrain_btn.clicked.connect(self._on_clicked)
        btn_row.addWidget(self._retrain_btn)
        btn_row.addStretch(1)
        ac_lay.addLayout(btn_row)

        # Indeterminate progress strip — hidden until running
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            f"QProgressBar {{ border: none; background: {HAIR};"
            f" height: 4px; border-radius: 0; }}"
            f"QProgressBar::chunk {{ background: {ACCENT}; border-radius: 0; }}"
        )
        self._progress.setVisible(False)
        ac_lay.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            f"font-size: 11px; color: {TEXT_HINT};"
            f" font-family: '{FONT_MONO}', monospace;"
            " background: transparent; border: none;"
        )
        ac_lay.addWidget(self._status_lbl)

        body_lay.addWidget(action_card)

        # Results card
        self._results_card = GlassCard(radius=0, bg=BG_SURFACE, border_color=HAIR)
        rc_outer = QVBoxLayout(self._results_card)
        rc_outer.setContentsMargins(28, 20, 28, 20)
        rc_outer.setSpacing(12)

        rc_outer.addWidget(SectionLabel("Last Training Result"))

        # Top stats row
        self._stats_row = QHBoxLayout()
        self._stats_row.setSpacing(20)
        self._stat_acc = self._make_stat("Accuracy", "—")
        self._stat_n   = self._make_stat("Samples",  "—")
        self._stat_ts  = self._make_stat("Trained at", "—")
        self._stats_row.addWidget(self._stat_acc[0])
        self._stats_row.addWidget(self._stat_n[0])
        self._stats_row.addWidget(self._stat_ts[0], 1)
        rc_outer.addLayout(self._stats_row)

        rc_outer.addWidget(HSep())

        rc_outer.addWidget(SectionLabel("Per-sign Accuracy"))

        self._per_sign_scroll = QScrollArea()
        self._per_sign_scroll.setWidgetResizable(True)
        self._per_sign_scroll.setStyleSheet("border: none; background: transparent;")

        self._per_sign_inner = QWidget()
        self._per_sign_lay = QVBoxLayout(self._per_sign_inner)
        self._per_sign_lay.setContentsMargins(0, 0, 0, 0)
        self._per_sign_lay.setSpacing(2)
        self._per_sign_lay.addStretch(1)
        self._per_sign_scroll.setWidget(self._per_sign_inner)
        rc_outer.addWidget(self._per_sign_scroll, 1)

        body_lay.addWidget(self._results_card, 1)

    # ─── Stat helper ──────────────────────────────────────────────────────────
    def _make_stat(self, label: str, value: str) -> tuple[QWidget, QLabel]:
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        cap = QLabel(label.upper())
        cap.setStyleSheet(
            f"font-size: 9px; color: {TEXT_HINT};"
            f" font-family: '{FONT_MONO}', monospace; letter-spacing: 1.5px;"
            " background: transparent; border: none;"
        )
        val = QLabel(value)
        val.setStyleSheet(
            f"font-size: 18px; color: {TEXT_TITLE};"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            " background: transparent; border: none;"
        )
        lay.addWidget(cap)
        lay.addWidget(val)
        return wrap, val

    # ─── Public API ───────────────────────────────────────────────────────────
    def show_running(self) -> None:
        """Switch to the running state."""
        self._retrain_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status_lbl.setText("Training in progress…")
        self._status_lbl.setStyleSheet(
            f"font-size: 11px; color: {ACCENT};"
            f" font-family: '{FONT_MONO}', monospace;"
            " background: transparent; border: none;"
        )

    def show_results(self, meta: dict) -> None:
        """Populate the results panel from a training_meta.json dict."""
        self._retrain_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_lbl.setText("Training complete.")
        self._status_lbl.setStyleSheet(
            f"font-size: 11px; color: {SUCCESS};"
            f" font-family: '{FONT_MONO}', monospace;"
            " background: transparent; border: none;"
        )

        acc = float(meta.get("overall_accuracy", 0.0))
        n   = int(meta.get("num_samples", 0))
        ts  = str(meta.get("trained_at", "—"))
        self._stat_acc[1].setText(f"{acc * 100:.1f}%")
        self._stat_n[1].setText(f"{n}")
        self._stat_ts[1].setText(ts.replace("T", "  "))

        self._populate_per_sign(meta.get("per_sign", {}))

    def show_error(self, msg: str) -> None:
        self._retrain_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_lbl.setText(f"Error: {msg}")
        self._status_lbl.setStyleSheet(
            f"font-size: 11px; color: {DANGER};"
            f" font-family: '{FONT_MONO}', monospace;"
            " background: transparent; border: none;"
        )

    def populate_from_meta(self, meta: dict) -> None:
        """Render the results card from an existing meta dict (no status change)."""
        if not meta:
            return
        acc = float(meta.get("overall_accuracy", 0.0))
        n   = int(meta.get("num_samples", 0))
        ts  = str(meta.get("trained_at", "—"))
        self._stat_acc[1].setText(f"{acc * 100:.1f}%")
        self._stat_n[1].setText(f"{n}")
        self._stat_ts[1].setText(ts.replace("T", "  "))
        self._populate_per_sign(meta.get("per_sign", {}))

    # ─── Internals ────────────────────────────────────────────────────────────
    def _populate_per_sign(self, per_sign: dict) -> None:
        # Clear existing rows
        while self._per_sign_lay.count() > 1:
            item = self._per_sign_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not per_sign:
            empty = QLabel("(no per-sign data)")
            empty.setStyleSheet(
                f"font-size: 11px; color: {TEXT_HINT};"
                f" font-family: '{FONT_MONO}', monospace;"
                " background: transparent; border: none;"
            )
            self._per_sign_lay.insertWidget(0, empty)
            return

        for sign in sorted(per_sign.keys()):
            entry = per_sign[sign]
            acc   = float(entry.get("accuracy", 0.0))
            sup   = int(entry.get("support", 0))

            row = QFrame()
            row.setStyleSheet(
                f"QFrame {{ background: transparent; border: none; }}"
                f"QLabel {{ background: transparent; border: none; }}"
            )
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 4, 0, 4)
            row_lay.setSpacing(8)

            name = QLabel(sign.replace("_", " ").upper())
            name.setStyleSheet(
                f"font-size: 12px; color: {TEXT_PRIMARY};"
                f" font-family: '{FONT_MONO}', monospace; letter-spacing: 1px;"
            )
            row_lay.addWidget(name, 1)

            sup_lbl = QLabel(f"n={sup}")
            sup_lbl.setStyleSheet(
                f"font-size: 10px; color: {TEXT_HINT};"
                f" font-family: '{FONT_MONO}', monospace;"
            )
            row_lay.addWidget(sup_lbl)

            color = SUCCESS if acc >= 0.85 else (WARNING if acc >= 0.65 else DANGER)
            acc_lbl = QLabel(f"{acc * 100:.0f}%")
            acc_lbl.setFixedWidth(56)
            acc_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            acc_lbl.setStyleSheet(
                f"font-size: 12px; font-weight: 600; color: {color};"
                f" font-family: '{FONT_MONO}', monospace;"
            )
            row_lay.addWidget(acc_lbl)

            self._per_sign_lay.insertWidget(self._per_sign_lay.count() - 1, row)

    def _on_clicked(self) -> None:
        self.retrain_requested.emit()
