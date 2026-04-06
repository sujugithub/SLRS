"""
Home screen — PyQt6.
Hero section, two CardButtons, signs list with delete.
"""
from __future__ import annotations

import os
import shutil

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QVBoxLayout, QWidget,
)

from config import PRETRAINED_DATA_DIR, CUSTOM_DATA_DIR, DEFAULT_MODEL_FILE
from gui.design import (
    ACCENT, BG_BASE, BG_BORDER, BG_DEEP, BG_ELEVATED, BG_SURFACE,
    DANGER, SUCCESS, TEXT_HINT, TEXT_PRIMARY, TEXT_SEC, TEXT_TITLE,
    HAIR,
    CardButton, DangerButton, GlassCard, HSep, PillButton,
    SectionLabel, StatusDot,
)


class HomeScreen(QWidget):
    """Landing screen with hero, navigation cards, and signs list."""

    def __init__(self, on_train_click=None, on_predict_click=None,
                 parent=None):
        super().__init__(parent)
        self.on_train_click   = on_train_click
        self.on_predict_click = on_predict_click
        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background-color: {BG_DEEP};")

        # ── Hero ─────────────────────────────────────────────────────────
        hero = QWidget()
        hero.setFixedHeight(148)
        hero.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-bottom: 1px solid {HAIR};")
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(0, 20, 0, 20)
        hero_lay.setSpacing(6)
        hero_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Sign Language AI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 32px; font-weight: 400; color: {TEXT_TITLE};"
            f" font-family: Georgia, 'Times New Roman', serif;"
            " background: transparent; border: none; letter-spacing: -1px;")
        hero_lay.addWidget(title)

        subtitle = QLabel(
            "real-time recognition  ·  custom training  ·  ai sentence builder")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            f"font-size: 11px; color: {TEXT_HINT};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        hero_lay.addWidget(subtitle)

        root.addWidget(hero)

        # ── Body ─────────────────────────────────────────────────────────
        body = QWidget()
        body.setStyleSheet(f"background-color: {BG_DEEP};")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(28, 24, 28, 24)
        body_lay.setSpacing(20)
        root.addWidget(body, 1)

        # Navigation cards
        nav_row = QHBoxLayout()
        nav_row.setSpacing(12)

        train_card = CardButton(
            title="Train Signs",
            subtitle="Record gestures and build your model",
            icon="✏️",
            accent=ACCENT,
        )
        train_card.clicked.connect(self._on_train)
        nav_row.addWidget(train_card, 1)

        detect_card = CardButton(
            title="Detect Signs",
            subtitle="Live camera recognition with sentence builder",
            icon="👁",
            accent=SUCCESS,
        )
        detect_card.clicked.connect(self._on_predict)
        nav_row.addWidget(detect_card, 1)
        body_lay.addLayout(nav_row)

        body_lay.addWidget(HSep())

        # Signs list panel
        signs_hdr = QHBoxLayout()
        signs_hdr.setSpacing(8)
        self._signs_sec_lbl = SectionLabel("Available Signs  —  0 signs")
        signs_hdr.addWidget(self._signs_sec_lbl, 1)
        delete_btn = DangerButton("✕  DELETE SELECTED")
        delete_btn.setFixedWidth(160)
        delete_btn.clicked.connect(self._delete_selected_sign)
        signs_hdr.addWidget(delete_btn)
        body_lay.addLayout(signs_hdr)

        self._signs_list = QListWidget()
        self._signs_list.setMinimumHeight(120)
        body_lay.addWidget(self._signs_list, 1)

        # Status row
        status_row = QHBoxLayout()
        self._status_dot = StatusDot(color=SUCCESS, size=7)
        status_row.addWidget(self._status_dot)
        status_row.addSpacing(6)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_SEC};"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            " background: transparent; border: none;")
        status_row.addWidget(self._status_lbl, 1)
        body_lay.addLayout(status_row)

    # ── Public API ───────────────────────────────────────────────────────────
    def activate(self):
        self.refresh_sign_list()

    def deactivate(self):
        pass

    # ── Signs list ───────────────────────────────────────────────────────────
    def refresh_sign_list(self):
        self._signs_list.clear()
        signs = set()
        for base_dir in (PRETRAINED_DATA_DIR, CUSTOM_DATA_DIR):
            if os.path.isdir(base_dir):
                for name in sorted(os.listdir(base_dir)):
                    if os.path.isdir(os.path.join(base_dir, name)):
                        signs.add(name)
        for sign in sorted(signs):
            self._signs_list.addItem(sign.replace("_", " ").title())
        count = len(signs)
        self._signs_sec_lbl.setText(
            f"Available Signs  —  {count} sign{'s' if count != 1 else ''}")
        if count:
            self._status_dot.set_color(SUCCESS)
            self._status_lbl.setText(f"{count} sign(s) loaded  ·  model ready")
        else:
            self._status_dot.set_color(TEXT_SEC)
            self._status_lbl.setText("No signs trained yet")

    def _delete_selected_sign(self):
        item = self._signs_list.currentItem()
        if not item:
            QMessageBox.information(
                self, "No Selection",
                "Please select a sign from the list to delete.")
            return
        sign_display = item.text()
        sign_key = sign_display.lower().replace(" ", "_")
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete all data for \"{sign_display}\"?\n\n"
            "This will remove the sign folder and invalidate the current model.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        deleted = False
        for base_dir in (PRETRAINED_DATA_DIR, CUSTOM_DATA_DIR):
            folder = os.path.join(base_dir, sign_key)
            if os.path.isdir(folder):
                try:
                    shutil.rmtree(folder)
                    deleted = True
                except OSError as e:
                    QMessageBox.critical(
                        self, "Delete Error",
                        f"Could not delete {folder}:\n{e}")
        if deleted:
            try:
                if os.path.isfile(DEFAULT_MODEL_FILE):
                    os.remove(DEFAULT_MODEL_FILE)
            except OSError:
                pass
            self.refresh_sign_list()
            self._status_lbl.setText(
                f"Deleted \"{sign_display}\"  ·  model reset")
            self._status_dot.set_color(DANGER)
        else:
            QMessageBox.warning(
                self, "Not Found",
                f"Could not find data folder for \"{sign_display}\".")

    # ── Navigation ───────────────────────────────────────────────────────────
    def _on_train(self):
        if self.on_train_click:
            self.on_train_click()

    def _on_predict(self):
        if self.on_predict_click:
            self.on_predict_click()
