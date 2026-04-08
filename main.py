"""
Main entry — PyQt6 version, redesigned UI.

Launches directly into PredictionScreen. The four configuration / training
flows (Train, Library, Retrain, Settings) are reachable only via the
hamburger menu in the top-right; clicking it opens a slide-in modal drawer
that hosts the four sub-screens.
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime


# ── Ensure Qt can locate the platform plugins (cocoa on macOS) when running
#    as a plain Python script rather than a bundled application.
def _set_qt_plugin_path() -> None:
    try:
        import PyQt6 as _pyqt6
        _qt_root = os.path.join(os.path.dirname(_pyqt6.__file__), "Qt6")
        _plugins  = os.path.join(_qt_root, "plugins")
        _platforms = os.path.join(_plugins, "platforms")
        if os.path.isdir(_platforms):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _platforms
        if os.path.isdir(_plugins):
            os.environ["QT_PLUGIN_PATH"] = _plugins
    except Exception:
        pass

_set_qt_plugin_path()

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QStackedWidget, QVBoxLayout, QWidget,
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
    ACCENT, BG_DEEP, BG_BASE, BG_SURFACE, HAIR, HAIR_STRONG,
    SUCCESS, TEXT_TITLE, TEXT_SEC, TEXT_HINT,
    FONT_DISPLAY, FONT_MONO, IconButton, PulsingDot,
    init_fonts, rebuild_global_qss,
)
from gui.menu_overlay import MenuDrawer
from gui.prediction_screen import PredictionScreen
from gui.train_dialog import TrainDialog
from gui.view_dialog import ViewDialog
from gui.retrain_dialog import RetrainDialog
from gui.settings_dialog import SettingsDialog


# ── Background workers ───────────────────────────────────────────────────────

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


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(max(WINDOW_WIDTH, 1280), max(WINDOW_HEIGHT, 800))
        self.setMinimumSize(1180, 760)

        # ── Central layout: top bar + content ──────────────────────────────
        central = QWidget()
        central.setStyleSheet(f"background-color: {BG_DEEP};")
        central_lay = QVBoxLayout(central)
        central_lay.setContentsMargins(0, 0, 0, 0)
        central_lay.setSpacing(0)
        self.setCentralWidget(central)

        # Top bar
        top_bar = QWidget()
        top_bar.setFixedHeight(56)
        top_bar.setStyleSheet(
            f"background-color: {BG_BASE};"
            f" border-bottom: 1px solid {HAIR};"
        )
        tb_lay = QHBoxLayout(top_bar)
        tb_lay.setContentsMargins(28, 0, 18, 0)
        tb_lay.setSpacing(0)

        # Brand mark
        mark = QLabel("◆")
        mark.setStyleSheet(
            f"color: {ACCENT}; font-size: 16px; font-weight: 700;"
            f" background: transparent; border: none;"
        )
        tb_lay.addWidget(mark)
        tb_lay.addSpacing(10)

        brand_col = QVBoxLayout()
        brand_col.setContentsMargins(0, 0, 0, 0)
        brand_col.setSpacing(0)

        brand_top = QLabel("SignBridge")
        brand_top.setStyleSheet(
            f"color: {TEXT_TITLE}; font-size: 15px; font-weight: 700;"
            f" font-family: '{FONT_DISPLAY}', sans-serif;"
            f" letter-spacing: -0.2px;"
            f" background: transparent; border: none;"
        )
        brand_sub = QLabel("LIVE SIGN RECOGNITION")
        brand_sub.setStyleSheet(
            f"color: {TEXT_HINT}; font-size: 8px; font-weight: 700;"
            f" font-family: '{FONT_MONO}', monospace; letter-spacing: 2.5px;"
            f" background: transparent; border: none;"
        )
        brand_col.addWidget(brand_top)
        brand_col.addWidget(brand_sub)
        tb_lay.addLayout(brand_col)

        tb_lay.addStretch(1)

        # Right side: status pill + hamburger
        self._status_dot = PulsingDot(
            on_color=ACCENT, off_color="#0a3a35", size=8,
        )
        tb_lay.addWidget(self._status_dot)
        tb_lay.addSpacing(8)

        mediapipe_lbl = QLabel("MEDIAPIPE ACTIVE")
        mediapipe_lbl.setStyleSheet(
            f"color: {TEXT_SEC}; font-size: 9px;"
            f" font-family: '{FONT_MONO}', monospace; letter-spacing: 2px;"
            f" font-weight: 700;"
            f" background: transparent; border: none;"
        )
        tb_lay.addWidget(mediapipe_lbl)
        tb_lay.addSpacing(20)

        # Vertical separator
        sep = QWidget()
        sep.setFixedSize(1, 24)
        sep.setStyleSheet(f"background-color: {HAIR_STRONG}; border: none;")
        tb_lay.addWidget(sep)
        tb_lay.addSpacing(12)

        # Hamburger
        self._menu_btn = IconButton("☰", size=40)
        self._menu_btn.clicked.connect(self._open_menu)
        tb_lay.addWidget(self._menu_btn)

        central_lay.addWidget(top_bar)

        # ── Stacked widget for splash → prediction screen ──────────────────
        self._stack = QStackedWidget()
        central_lay.addWidget(self._stack, 1)

        splash = QLabel("◆  INITIALIZING SIGNBRIDGE…")
        splash.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {ACCENT};"
            f" font-family: '{FONT_MONO}', monospace; letter-spacing: 3px;"
            f" background-color: {BG_DEEP};"
        )
        splash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(splash)
        self._stack.setCurrentWidget(splash)
        QApplication.processEvents()

        # ── Settings + phrases ─────────────────────────────────────────────
        self._settings = settings_store.load_settings()
        self._phrase_matcher = phrase_store.PhraseMatcher(
            phrase_store.load_phrases())

        # Shared resources — built using settings values.
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

        # ── Build prediction screen (always-on main view) ──────────────────
        self.prediction_screen = PredictionScreen(
            model=self.model,
            on_back=None,
            camera=self.camera,
            detector=self.detector,
            pose_detector=self.pose_detector,
            speaker=self.speaker,
            holistic=self.holistic,
            lstm_model=self.lstm_model,
        )
        self.prediction_screen.apply_settings(self._settings)
        self.prediction_screen.set_phrase_matcher(self._phrase_matcher)
        self._stack.addWidget(self.prediction_screen)

        # ── Build sub-screens (hosted inside the drawer) ───────────────────
        self.train_dialog = TrainDialog(
            camera=self.camera,
            holistic=self.holistic,
        )
        self.train_dialog.save_requested.connect(self._on_train_save)
        self.train_dialog.back_requested.connect(self._close_drawer)

        self.view_dialog = ViewDialog()
        self.view_dialog.add_samples_requested.connect(self._on_view_add_samples)
        self.view_dialog.delete_requested.connect(self._on_view_delete)
        self.view_dialog.back_requested.connect(self._close_drawer)

        self.retrain_dialog = RetrainDialog()
        self.retrain_dialog.retrain_requested.connect(self._on_retrain_clicked)
        self.retrain_dialog.back_requested.connect(self._close_drawer)
        self.retrain_dialog.populate_from_meta(training_meta.load_meta() or {})

        self.settings_dialog = SettingsDialog()
        self.settings_dialog.settings_changed.connect(self._on_settings_saved)
        self.settings_dialog.back_requested.connect(self._close_drawer)

        # ── Build menu drawer (overlay child of MainWindow) ────────────────
        self._menu = MenuDrawer(self)
        self._menu.add_screen("train",    self.train_dialog)
        self._menu.add_screen("view",     self.view_dialog)
        self._menu.add_screen("retrain",  self.retrain_dialog)
        self._menu.add_screen("settings", self.settings_dialog)
        self._menu.item_selected.connect(self._on_menu_item)
        self._menu.closed.connect(self._on_menu_closed)

        # State trackers
        self._rf_worker      = None
        self._retrain_worker = None
        self._current_drawer_key: str | None = None

        # Switch to prediction screen
        self._stack.removeWidget(splash)
        splash.deleteLater()
        self._stack.setCurrentWidget(self.prediction_screen)
        self.prediction_screen.activate()

    # ── Hamburger drawer plumbing ────────────────────────────────────────────
    def _open_menu(self) -> None:
        # Pause the prediction screen so it releases the camera before any
        # sub-screen (TrainDialog) might claim it.
        try:
            self.prediction_screen.deactivate()
        except Exception:
            pass
        self._menu.open_drawer(initial_key="train")

    def _close_drawer(self) -> None:
        self._menu.close_drawer()

    def _on_menu_closed(self) -> None:
        # Tear down the active sub-screen and resume prediction.
        self._deactivate_current_sub()
        self._current_drawer_key = None
        try:
            self.prediction_screen.activate()
        except Exception:
            pass

    def _on_menu_item(self, key: str) -> None:
        # Deactivate the previously visible sub-screen, activate the new one.
        if self._current_drawer_key == key:
            return
        self._deactivate_current_sub()
        self._current_drawer_key = key
        self._activate_sub(key)

    def _activate_sub(self, key: str) -> None:
        if key == "train":
            self.train_dialog.activate()
        elif key == "view":
            self.view_dialog.refresh()
        elif key == "retrain":
            try:
                self.retrain_dialog.populate_from_meta(
                    training_meta.load_meta() or {})
            except Exception:
                pass
        elif key == "settings":
            self.settings_dialog.load_into_ui(self._settings)

    def _deactivate_current_sub(self) -> None:
        key = self._current_drawer_key
        if key == "train":
            try:
                self.train_dialog.deactivate()
            except Exception:
                pass

    # ── Train flow ───────────────────────────────────────────────────────────
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
            QMessageBox.critical(
                self, "Save Error", f"Failed to save training data:\n{msg}")

    # ── View flow ────────────────────────────────────────────────────────────
    def _on_view_add_samples(self, sign_name: str) -> None:
        # Switch to the train sub-screen and prefill with the sign name.
        self._deactivate_current_sub()
        self._current_drawer_key = "train"
        self._menu.select("train")
        self.train_dialog.activate(prefill=sign_name)

    def _on_view_delete(self, sign_name: str) -> None:
        sign_dir = os.path.join(CUSTOM_DATA_DIR, sign_name)
        try:
            shutil.rmtree(sign_dir, ignore_errors=True)
        except Exception as exc:
            QMessageBox.critical(
                self, "Delete Error", f"Could not delete folder:\n{exc}")
            return
        try:
            self.model.evict_sign(sign_name)
            self.model.train()
            self.model.save_model()
        except Exception as exc:
            print(f"[main] model evict/retrain after delete failed: {exc}")
        self.view_dialog.refresh()

    # ── Retrain flow ─────────────────────────────────────────────────────────
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

    # ── Settings flow ────────────────────────────────────────────────────────
    def _on_settings_saved(self, new_settings: dict) -> None:
        old_camera_idx = self._settings.get("camera_index")
        old_mp_complexity = self._settings.get("mediapipe_complexity")

        self._settings.update(new_settings)
        settings_store.save_settings(self._settings)

        if new_settings.get("camera_index") != old_camera_idx:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = CameraHandler(
                camera_index=new_settings["camera_index"])
            self.prediction_screen.set_camera(self.camera)
            self.train_dialog._camera = self.camera

        if new_settings.get("mediapipe_complexity") != old_mp_complexity:
            try:
                self.holistic.close()
            except Exception:
                pass
            self.holistic = HolisticDetector(
                model_complexity=new_settings["mediapipe_complexity"])
            self.prediction_screen.set_holistic(self.holistic)
            self.train_dialog._holistic = self.holistic

        self.prediction_screen.apply_settings(self._settings)

        self._close_drawer()

    # ── Resize → keep drawer geometry in sync ────────────────────────────────
    def resizeEvent(self, event):
        if self._menu is not None and self._menu.isVisible():
            self._menu.setGeometry(self.centralWidget().rect())
        super().resizeEvent(event)

    # ── Close ────────────────────────────────────────────────────────────────
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


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    # Resolve fonts now that QApplication exists, then build the stylesheet.
    init_fonts()
    qss = rebuild_global_qss()
    app.setStyleSheet(qss)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
