
"""
Main entry — PyQt6 version.
QMainWindow + QStackedWidget, QThread workers for training.
"""
from __future__ import annotations

import os
import sys

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

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QGraphicsOpacityEffect, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget,
)

from config import (
    APP_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    CUSTOM_DATA_DIR, SEQUENCE_DATA_DIR, SEQ_LENGTH,
    LSTM_MODEL_FILE, LSTM_LABELS_FILE,
)
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
from gui.home_screen import HomeScreen
from gui.training_screen import TrainingScreen
from gui.prediction_screen import PredictionScreen


# ── Background workers ─────────────────────────────────────────────────────────

class RFTrainWorker(QThread):
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
                        # Old data has a different feature width (e.g. 63 legacy
                        # vs 52 spatial). Discard the stale file so the model
                        # retrains cleanly on the new format.
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


class LSTMTrainWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, lstm_model, sign_name: str, sequences_list: list,
                 parent=None):
        super().__init__(parent)
        self._lstm_model    = lstm_model
        self._sign_name     = sign_name
        self._sequences_list = sequences_list

    def run(self):
        try:
            sign_seq_dir = os.path.join(SEQUENCE_DATA_DIR, self._sign_name)
            os.makedirs(sign_seq_dir, exist_ok=True)
            new_seqs = np.array(self._sequences_list, dtype=np.float32)
            seq_path = os.path.join(sign_seq_dir, "sequences.npy")
            if os.path.isfile(seq_path):
                try:
                    existing = np.load(seq_path)
                    new_seqs = np.concatenate([existing, new_seqs], axis=0)
                except Exception:
                    pass
            np.save(seq_path, new_seqs)

            all_sequences: dict = {}
            if os.path.isdir(SEQUENCE_DATA_DIR):
                for sname in os.listdir(SEQUENCE_DATA_DIR):
                    sdir  = os.path.join(SEQUENCE_DATA_DIR, sname)
                    spath = os.path.join(sdir, "sequences.npy")
                    if os.path.isfile(spath):
                        try:
                            all_sequences[sname] = np.load(spath)
                        except Exception:
                            pass

            if not all_sequences:
                raise ValueError("No sequence data found to train on.")

            result = self._lstm_model.train(all_sequences)
            self._lstm_model.save(LSTM_MODEL_FILE, LSTM_LABELS_FILE)
            self.finished.emit({
                "sign_name":     self._sign_name,
                "accuracy":      result.get("accuracy", 0),
                "num_sequences": result.get("num_sequences", 0),
            })
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
            f" border-bottom: 1px solid {HAIR};")
        tb_lay = QHBoxLayout(top_bar)
        tb_lay.setContentsMargins(20, 0, 20, 0)
        tb_lay.setSpacing(0)

        brand = QLabel("SignBridge")
        brand.setStyleSheet(
            f"color: {TEXT_TITLE}; font-size: 13px; font-weight: 600;"
            f" font-family: Georgia, 'Times New Roman', serif;"
            f" background: transparent; border: none; letter-spacing: 1px;")
        tb_lay.addWidget(brand)
        tb_lay.addStretch(1)

        self._status_dot = PulsingDot(on_color=SUCCESS, off_color="#2a4a35", size=7)
        tb_lay.addWidget(self._status_dot)
        tb_lay.addSpacing(8)

        mediapipe_lbl = QLabel("MediaPipe Active")
        mediapipe_lbl.setStyleSheet(
            f"color: {TEXT_SEC}; font-size: 10px;"
            f" font-family: 'Courier New', monospace; letter-spacing: 1px;"
            f" background: transparent; border: none;")
        tb_lay.addWidget(mediapipe_lbl)

        central_lay.addWidget(top_bar)

        # Stacked widget goes below the top bar
        self._stack = QStackedWidget()
        central_lay.addWidget(self._stack, 1)

        # Shared resources
        self.camera       = CameraHandler()
        self.detector     = HandDetector()
        self.pose_detector = PoseDetector()
        self.holistic     = HolisticDetector()
        self.model        = SignModel()
        self.lstm_model   = LSTMSignModel(seq_len=SEQ_LENGTH)
        self.speaker      = TTSSpeaker()

        # Loading splash
        splash = QLabel("◆  Initializing SignAI…")
        splash.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {ACCENT};"
            f" background-color: {BG_DEEP};")
        splash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(splash)
        self._stack.setCurrentWidget(splash)
        QApplication.processEvents()

        self.model.load_pretrained()
        self.lstm_model.load(LSTM_MODEL_FILE, LSTM_LABELS_FILE)

        # Build screens
        self.home_screen = HomeScreen(
            on_train_click=self.show_training,
            on_predict_click=self.show_prediction,
        )
        self.training_screen = TrainingScreen(
            on_back=self.show_home,
            on_save=self._on_training_save,
            on_save_sequences=self._on_sequences_save,
            camera=self.camera,
            detector=self.detector,
            pose_detector=self.pose_detector,
            holistic=self.holistic,
        )
        self.prediction_screen = PredictionScreen(
            model=self.model,
            on_back=self.show_home,
            camera=self.camera,
            detector=self.detector,
            pose_detector=self.pose_detector,
            speaker=self.speaker,
            holistic=self.holistic,
            lstm_model=self.lstm_model,
        )

        self._stack.addWidget(self.home_screen)
        self._stack.addWidget(self.training_screen)
        self._stack.addWidget(self.prediction_screen)

        # Remove splash, go to home
        self._stack.removeWidget(splash)
        splash.deleteLater()

        self._current_screen = None
        self._rf_worker   = None
        self._lstm_worker = None

        self._switch_to(self.home_screen, animate=False)
        self.home_screen.activate()

    # ── Screen switching ───────────────────────────────────────────────────────
    def _switch_to(self, widget: QWidget, animate: bool = True):
        if self._current_screen and self._current_screen is not widget:
            self._current_screen.deactivate()

        self._stack.setCurrentWidget(widget)
        self._current_screen = widget

        if animate:
            eff = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(eff)
            anim = QPropertyAnimation(eff, b"opacity", widget)
            anim.setDuration(220)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.finished.connect(lambda: widget.setGraphicsEffect(None))
            anim.start()

    def show_home(self):
        self._switch_to(self.home_screen)
        self.home_screen.activate()

    def show_training(self):
        self._switch_to(self.training_screen)
        self.training_screen.reset()
        self.training_screen.activate()

    def show_prediction(self):
        self._switch_to(self.prediction_screen)
        self.prediction_screen.activate()

    # ── Training callbacks ─────────────────────────────────────────────────────
    def _on_training_save(self, sign_name: str, features_list: list):
        self.training_screen.show_training_overlay("Training model…")
        self._rf_worker = RFTrainWorker(self.model, sign_name, features_list, self)
        self._rf_worker.finished.connect(self._rf_done)
        self._rf_worker.error.connect(self._rf_error)
        self._rf_worker.start()

    def _rf_done(self, result: dict):
        self.training_screen.hide_training_overlay()
        sign  = result["sign_name"]
        saved = result["num_saved"]
        total = result["num_samples"]
        acc   = result["accuracy"]
        QMessageBox.information(
            self, "Training Complete",
            f"Sign \"{sign}\" saved!\n\n"
            f"Samples saved: {saved}\n"
            f"Total training data: {total}\n"
            f"Model accuracy: {acc:.1%}")
        self.show_home()

    def _rf_error(self, msg: str):
        self.training_screen.hide_training_overlay()
        QMessageBox.critical(self, "Save Error",
                             f"Failed to save training data:\n{msg}")
        self.show_home()

    def _on_sequences_save(self, sign_name: str, sequences_list: list):
        self.training_screen.show_training_overlay("Training LSTM…")
        self._lstm_worker = LSTMTrainWorker(
            self.lstm_model, sign_name, sequences_list, self)
        self._lstm_worker.finished.connect(self._lstm_done)
        self._lstm_worker.error.connect(self._lstm_error)
        self._lstm_worker.start()

    def _lstm_done(self, result: dict):
        self.training_screen.hide_training_overlay()
        sign = result["sign_name"]
        n    = result["num_sequences"]
        acc  = result["accuracy"]
        QMessageBox.information(
            self, "LSTM Training Complete",
            f"Dynamic sign \"{sign}\" saved!\n\n"
            f"Sequences used: {n}\n"
            f"Model accuracy: {acc:.1%}")
        self.show_home()

    def _lstm_error(self, msg: str):
        self.training_screen.hide_training_overlay()
        QMessageBox.critical(self, "LSTM Save Error",
                             f"Failed to save dynamic sign data:\n{msg}")
        self.show_home()

    # ── Close ──────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self.training_screen.has_unsaved_data():
            reply = QMessageBox.question(
                self, "Unsaved Data",
                "You have unsaved training captures.\nAre you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        self.training_screen.cleanup()
        self.prediction_screen.cleanup()

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
