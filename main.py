
"""
Main entry — PyQt6 version.

Launches directly into PredictionScreen. The four configuration / training
flows (Train, View, Retrain, Settings) are reachable only via a discreet ☰
hamburger menu in the top-right of the main window.
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime

# ── Ensure Qt can locate the platform plugins (cocoa on macOS) when running
#    as a plain Python script rather than a bundled application.
def _set_qt_plugin_path() -> None:
    """Set QT_QPA_PLATFORM_PLUGIN_PATH to the PyQt6-bundled platforms directory.

    On macOS, Qt's runtime plugin scanner requires QT_QPA_PLATFORM_PLUGIN_PATH
    to be set explicitly when running from a venv or non-standard Python install
    (e.g. uv-managed Python, pyenv, conda).  This must be done BEFORE any
    PyQt6.QtWidgets import triggers QApplication initialisation.
    """
    try:
        import PyQt6 as _pyqt6
        _qt_root = os.path.join(os.path.dirname(_pyqt6.__file__), "Qt6")
        _plugins  = os.path.join(_qt_root, "plugins")
        _platforms = os.path.join(_plugins, "platforms")

        # Always set (not setdefault) so a stale env var from a prior run
        # or from the shell cannot point Qt at the wrong directory.
        if os.path.isdir(_platforms):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _platforms
        if os.path.isdir(_plugins):
            os.environ["QT_PLUGIN_PATH"] = _plugins
    except Exception:
        pass

_set_qt_plugin_path()

import numpy as np

from PyQt6.QtCore import QPoint, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from config import (
    APP_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    CUSTOM_DATA_DIR, SEQUENCE_DATA_DIR, SEQ_LENGTH,
    LSTM_MODEL_FILE, LSTM_LABELS_FILE,
)
from core import settings_store, phrase_store, training_meta
from core.camera_handler import CameraHandler
from core.hand_detector import HandDetector
from core.pose_detector import PoseDetector
from core.holistic_detector import HolisticDetector
from core.model_trainer import SignModel
from core.lstm_trainer import LSTMSignModel
from core.tts_speaker import TTSSpeaker
from gui.design import (
    ACCENT, BG_DEEP, BG_BASE, HAIR,
    SUCCESS, TEXT_TITLE, TEXT_SEC,
    GLOBAL_QSS, PulsingDot,
)
from gui.menu_overlay import HamburgerMenu
from gui.prediction_screen import PredictionScreen
from gui.train_dialog import TrainDialog
from gui.view_dialog import ViewDialog
from gui.retrain_dialog import RetrainDialog
from gui.settings_dialog import SettingsDialog


# ── Background workers ─────────────────────────────────────────────────────────

class RFTrainWorker(QThread):
    """Saves freshly captured features and retrains the RF model."""

    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, model, sign_name: str, features_list: list, parent=None):
        super().__init__(parent)
        self._model        = model
        self._sign_name    = sign_name
        self._features_list = features_list

    def run(self):
        try:
            features_array = np.array(self._features_list, dtype=np.float64)
            sign_dir = os.path.join(CUSTOM_DATA_DIR, self._sign_name)
            os.makedirs(sign_dir, exist_ok=True)
            npy_path = os.path.join(sign_dir, "features.npy")
            if os.path.isfile(npy_path):
                try:
                    existing = np.load(npy_path)
                    if existing.shape[1] == features_array.shape[1]:
                        features_array = np.vstack([existing, features_array])
                    else:
                        # Old data has a different feature width — discard the
                        # stale file so the model retrains cleanly on the new
                        # format.
                        print(
                            f"[RFTrainWorker] Feature width changed "
                            f"({existing.shape[1]} → {features_array.shape[1]}) "
                            f"— replacing stale data for '{self._sign_name}'"
                        )
                except Exception:
                    pass
            np.save(npy_path, features_array)
            self._model.add_training_data(self._sign_name, features_array)
            result = self._model.train()
            self._model.save_model()
            self.finished.emit({
                "sign_name":    self._sign_name,
                "num_saved":    int(features_array.shape[0]),
                "accuracy":     result.get("accuracy", 0) if result else 0,
                "num_samples":  result.get("num_samples", 0) if result else 0,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class RetrainWorker(QThread):
    """Reloads every sign from disk and retrains the RF model from scratch."""

    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

    def run(self):
        try:
            self._model.reset_dataset()
            self._model._load_from_dir(CUSTOM_DATA_DIR)
            result = self._model.train() or {}
            self._model.save_model()
            meta = {
                "trained_at":       datetime.now().isoformat(timespec="seconds"),
                "overall_accuracy": float(result.get("accuracy", 0.0)),
                "num_samples":      int(result.get("num_samples", 0)),
                "per_sign":         result.get("per_sign", {}),
            }
            training_meta.save_meta(meta)
            self.finished.emit(meta)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(max(WINDOW_WIDTH, 1100), max(WINDOW_HEIGHT, 700))
        self.setMinimumSize(1100, 700)

        # ── Central layout: top bar + stacked content ──────────────────────
        central = QWidget()
        central.setStyleSheet(f"background-color: {BG_DEEP};")
        central_lay = QVBoxLayout(central)
        central_lay.setContentsMargins(0, 0, 0, 0)
        central_lay.setSpacing(0)
        self.setCentralWidget(central)

        # Slim 40px top bar
        top_bar = QWidget()
        top_bar.setFixedHeight(40)
        top_bar.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-bottom: 1px solid {HAIR};"
        )
        tb_lay = QHBoxLayout(top_bar)
        tb_lay.setContentsMargins(20, 0, 20, 0)
        tb_lay.setSpacing(0)

        brand = QLabel("SignBridge")
        brand.setStyleSheet(
            f"color: {TEXT_TITLE}; font-size: 13px; font-weight: 600;"
            f" font-family: Georgia, 'Times New Roman', serif;"
            f" background: transparent; border: none; letter-spacing: 1px;"
        )
        tb_lay.addWidget(brand)
        tb_lay.addStretch(1)

        self._status_dot = PulsingDot(on_color=SUCCESS, off_color="#2a4a35", size=7)
        tb_lay.addWidget(self._status_dot)
        tb_lay.addSpacing(8)

        mediapipe_lbl = QLabel("MediaPipe Active")
        mediapipe_lbl.setStyleSheet(
            f"color: {TEXT_SEC}; font-size: 10px;"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            f" background: transparent; border: none;"
        )
        tb_lay.addWidget(mediapipe_lbl)
        tb_lay.addSpacing(12)

        # ☰ hamburger button — the only navigation surface
        self._menu_btn = QPushButton("\u2630")
        self._menu_btn.setFixedSize(28, 28)
        self._menu_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._menu_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_SEC};"
            f" border: none; font-size: 16px; }}"
            f"QPushButton:hover {{ color: {TEXT_TITLE}; }}"
        )
        self._menu_btn.clicked.connect(self._open_menu)
        tb_lay.addWidget(self._menu_btn)

        central_lay.addWidget(top_bar)

        # Stacked widget goes below the top bar
        self._stack = QStackedWidget()
        central_lay.addWidget(self._stack, 1)

        # ── Splash while we initialise heavy resources ─────────────────────
        splash = QLabel("\u25c6  Initializing SignBridge…")
        splash.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {ACCENT};"
            f" background-color: {BG_DEEP};"
        )
        splash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(splash)
        self._stack.setCurrentWidget(splash)
        QApplication.processEvents()

        # ── Settings + phrases (loaded before any camera/holistic init) ────
        self._settings = settings_store.load_settings()
        self._phrase_matcher = phrase_store.PhraseMatcher(
            phrase_store.load_phrases())

        # Shared resources — built using settings values so the user's
        # camera index / mediapipe complexity choices apply on first launch.
        self.camera = CameraHandler(camera_index=self._settings["camera_index"])
        self.detector = HandDetector()
        self.pose_detector = PoseDetector()
        self.holistic = HolisticDetector(
            model_complexity=self._settings["mediapipe_complexity"])
        self.model = SignModel()
        self.lstm_model = LSTMSignModel(seq_len=SEQ_LENGTH)
        self.speaker = TTSSpeaker()

        self.model.load_pretrained()
        self.lstm_model.load(LSTM_MODEL_FILE, LSTM_LABELS_FILE)

        # ── Build screens ──────────────────────────────────────────────────
        self.prediction_screen = PredictionScreen(
            model=self.model,
            on_back=None,                 # back button removed; menu replaces it
            camera=self.camera,
            detector=self.detector,
            pose_detector=self.pose_detector,
            speaker=self.speaker,
            holistic=self.holistic,
            lstm_model=self.lstm_model,
        )
        self.prediction_screen.apply_settings(self._settings)
        self.prediction_screen.set_phrase_matcher(self._phrase_matcher)

        self.train_dialog = TrainDialog(
            camera=self.camera,
            holistic=self.holistic,
        )
        self.train_dialog.save_requested.connect(self._on_train_save)
        self.train_dialog.back_requested.connect(self._show_prediction_default)

        self.view_dialog = ViewDialog()
        self.view_dialog.add_samples_requested.connect(self._on_view_add_samples)
        self.view_dialog.delete_requested.connect(self._on_view_delete)
        self.view_dialog.back_requested.connect(self._show_prediction_default)

        self.retrain_dialog = RetrainDialog()
        self.retrain_dialog.retrain_requested.connect(self._on_retrain_clicked)
        self.retrain_dialog.back_requested.connect(self._show_prediction_default)
        # Pre-populate the results panel from any prior retrain
        self.retrain_dialog.populate_from_meta(training_meta.load_meta() or {})

        self.settings_dialog = SettingsDialog()
        self.settings_dialog.settings_changed.connect(self._on_settings_saved)
        self.settings_dialog.back_requested.connect(self._show_prediction_default)

        self._stack.addWidget(self.prediction_screen)
        self._stack.addWidget(self.train_dialog)
        self._stack.addWidget(self.view_dialog)
        self._stack.addWidget(self.retrain_dialog)
        self._stack.addWidget(self.settings_dialog)

        # ── Hamburger popup ────────────────────────────────────────────────
        self._menu = HamburgerMenu(self)
        self._menu.item_selected.connect(self._on_menu_selected)

        # Remove splash, switch to prediction screen
        self._stack.removeWidget(splash)
        splash.deleteLater()

        self._current_screen = None
        self._rf_worker      = None
        self._retrain_worker = None
        self._pending_view_refresh = False

        self._switch_to(self.prediction_screen)
        self.prediction_screen.activate()

    # ── Screen switching ───────────────────────────────────────────────────────
    def _switch_to(self, widget: QWidget) -> None:
        if self._current_screen and self._current_screen is not widget:
            try:
                self._current_screen.deactivate()
            except Exception:
                pass
        self._stack.setCurrentWidget(widget)
        self._current_screen = widget

    def _show_prediction_default(self) -> None:
        self._switch_to(self.prediction_screen)
        self.prediction_screen.activate()
        if self._pending_view_refresh:
            self._pending_view_refresh = False

    # ── Hamburger menu ─────────────────────────────────────────────────────────
    def _open_menu(self) -> None:
        pos = self._menu_btn.mapToGlobal(QPoint(0, self._menu_btn.height() + 4))
        # Anchor by right edge so the popup doesn't extend off-screen.
        pos.setX(pos.x() - self._menu.width() + self._menu_btn.width())
        self._menu.show_at(pos)

    def _on_menu_selected(self, key: str) -> None:
        if key == "train":
            self._show_train()
        elif key == "view":
            self._show_view()
        elif key == "retrain":
            self._show_retrain()
        elif key == "settings":
            self._show_settings()

    # ── Page openers ───────────────────────────────────────────────────────────
    def _show_train(self, prefill: str | None = None) -> None:
        self._switch_to(self.train_dialog)
        self.train_dialog.activate(prefill=prefill)

    def _show_view(self) -> None:
        self._switch_to(self.view_dialog)
        self.view_dialog.refresh()

    def _show_retrain(self) -> None:
        self._switch_to(self.retrain_dialog)

    def _show_settings(self) -> None:
        self._switch_to(self.settings_dialog)
        self.settings_dialog.load_into_ui(self._settings)

    # ── Train flow ─────────────────────────────────────────────────────────────
    def _on_train_save(self, sign_name: str, features_list: list) -> None:
        self._rf_worker = RFTrainWorker(
            self.model, sign_name, features_list, self)
        self._rf_worker.finished.connect(self._rf_done)
        self._rf_worker.error.connect(self._rf_error)
        self._rf_worker.start()

    def _rf_done(self, result: dict) -> None:
        try:
            self.train_dialog.on_save_complete(result)
        except Exception as exc:
            print(f"[main] train_dialog.on_save_complete failed: {exc}")

    def _rf_error(self, msg: str) -> None:
        try:
            self.train_dialog.on_save_error(msg)
        except Exception:
            QMessageBox.critical(self, "Save Error",
                                 f"Failed to save training data:\n{msg}")

    # ── View flow ──────────────────────────────────────────────────────────────
    def _on_view_add_samples(self, sign_name: str) -> None:
        self._show_train(prefill=sign_name)

    def _on_view_delete(self, sign_name: str) -> None:
        sign_dir = os.path.join(CUSTOM_DATA_DIR, sign_name)
        try:
            shutil.rmtree(sign_dir, ignore_errors=True)
        except Exception as exc:
            QMessageBox.critical(self, "Delete Error",
                                 f"Could not delete folder:\n{exc}")
            return
        try:
            self.model.evict_sign(sign_name)
            self.model.train()
            self.model.save_model()
        except Exception as exc:
            print(f"[main] model evict/retrain after delete failed: {exc}")
        self.view_dialog.refresh()

    # ── Retrain flow ───────────────────────────────────────────────────────────
    def _on_retrain_clicked(self) -> None:
        if self._retrain_worker is not None and self._retrain_worker.isRunning():
            return
        self.retrain_dialog.show_running()
        self._retrain_worker = RetrainWorker(self.model, self)
        self._retrain_worker.finished.connect(self._retrain_done)
        self._retrain_worker.error.connect(self._retrain_error)
        self._retrain_worker.start()

    def _retrain_done(self, meta: dict) -> None:
        self.retrain_dialog.show_results(meta)

    def _retrain_error(self, msg: str) -> None:
        self.retrain_dialog.show_error(msg)

    # ── Settings flow ──────────────────────────────────────────────────────────
    def _on_settings_saved(self, new_settings: dict) -> None:
        old_camera_idx = self._settings.get("camera_index")
        old_mp_complexity = self._settings.get("mediapipe_complexity")

        self._settings.update(new_settings)
        settings_store.save_settings(self._settings)

        # Camera index changed → swap CameraHandler and rebuild the worker.
        if new_settings.get("camera_index") != old_camera_idx:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = CameraHandler(
                camera_index=new_settings["camera_index"])
            self.prediction_screen.set_camera(self.camera)
            self.train_dialog._camera = self.camera   # train dialog reuses camera

        # MediaPipe complexity changed → swap HolisticDetector.
        if new_settings.get("mediapipe_complexity") != old_mp_complexity:
            try:
                self.holistic.close()
            except Exception:
                pass
            self.holistic = HolisticDetector(
                model_complexity=new_settings["mediapipe_complexity"])
            self.prediction_screen.set_holistic(self.holistic)
            self.train_dialog._holistic = self.holistic

        # Apply threshold / smoothing / TTS / voice live.
        self.prediction_screen.apply_settings(self._settings)

        # Snap back to the prediction screen so the user sees the change.
        self._show_prediction_default()

    # ── Close ──────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self.train_dialog.has_unsaved_data():
            reply = QMessageBox.question(
                self, "Unsaved Data",
                "You have unsaved training captures.\n"
                "Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        try: self.train_dialog.cleanup()
        except Exception: pass
        try: self.prediction_screen.cleanup()
        except Exception: pass

        try: self.camera.stop()
        except Exception: pass
        try: self.detector.close()
        except Exception: pass
        try: self.pose_detector.close()
        except Exception: pass
        try: self.holistic.close()
        except Exception: pass
        try: self.speaker.stop()
        except Exception: pass

        event.accept()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
