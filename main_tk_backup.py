"""
Main entry point for the Sign Language Recognition App.

Initializes the application, sets up the main window, and coordinates
navigation between the home, training, and prediction screens.
"""

import os
import tkinter as tk
from tkinter import messagebox

import numpy as np

from config import (
    APP_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    CUSTOM_DATA_DIR, SEQUENCE_DATA_DIR, SEQ_LENGTH,
    LSTM_MODEL_FILE, LSTM_LABELS_FILE,
)
from gui.design import BG_DEEP, BG_ELEVATED, ACCENT, SUCCESS, TEXT_SEC, F_BODY, _f
from core.camera_handler import CameraHandler
from core.hand_detector import HandDetector
from core.pose_detector import PoseDetector
from core.holistic_detector import HolisticDetector
from core.model_trainer import SignModel
from core.lstm_trainer import LSTMSignModel
from core.tts_speaker import TTSSpeaker
from gui.home_screen import HomeScreen
from gui.training_screen import TrainingScreen
from gui.prediction_screen import PredictionScreen


class App:
    """Top-level application controller managing screen navigation."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=BG_DEEP)
        self.root.minsize(700, 500)

        # Shared resources — single camera, detector, and model for all screens.
        # One instance of each is created here and injected into screens that
        # need them, so the camera isn't opened twice and the detector doesn't
        # allocate duplicate MediaPipe sessions.
        self.camera = CameraHandler()
        self.detector = HandDetector()
        self.pose_detector = PoseDetector()
        self.holistic = HolisticDetector()
        self.model = SignModel()
        self.lstm_model = LSTMSignModel(seq_len=SEQ_LENGTH)
        self.speaker = TTSSpeaker()

        # Show a temporary loading label while the model loads
        self._loading_label = tk.Label(
            self.root,
            text="◆  Initializing SignAI…",
            font=_f(18, "bold"),
            fg=ACCENT,
            bg=BG_DEEP,
        )
        self._loading_label.pack(expand=True)
        tk.Label(
            self.root,
            text="Loading AI model",
            font=F_BODY,
            fg=TEXT_SEC,
            bg=BG_DEEP,
        ).pack()
        self.root.update_idletasks()

        self.model.load_pretrained()
        self.lstm_model.load(LSTM_MODEL_FILE, LSTM_LABELS_FILE)

        self._loading_label.destroy()

        self.home_screen = HomeScreen(
            self.root,
            on_train_click=self.show_training,
            on_predict_click=self.show_prediction,
        )
        self.training_screen = TrainingScreen(
            self.root,
            on_back=self.show_home,
            on_save=self._on_training_save,
            on_save_sequences=self._on_sequences_save,
            camera=self.camera,
            detector=self.detector,
            pose_detector=self.pose_detector,
            holistic=self.holistic,
        )
        self.prediction_screen = PredictionScreen(
            self.root,
            model=self.model,
            on_back=self.show_home,
            camera=self.camera,
            detector=self.detector,
            pose_detector=self.pose_detector,
            speaker=self.speaker,
            holistic=self.holistic,
            lstm_model=self.lstm_model,
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.home_screen.show()

    def show_home(self):
        """Switch to the home screen."""
        self.training_screen.hide()
        self.prediction_screen.hide()
        self.home_screen.refresh_sign_list()
        self.home_screen.show()

    def show_training(self):
        """Switch to the training screen."""
        self.home_screen.hide()
        self.prediction_screen.hide()
        self.training_screen.reset()
        self.training_screen.show()

    def show_prediction(self):
        """Switch to the prediction screen."""
        self.home_screen.hide()
        self.training_screen.hide()
        self.prediction_screen.show()

    def _on_training_save(self, sign_name, features_list):
        """Save captured features to disk, retrain the model, and go home."""
        # Show a blocking "Training..." overlay so the user knows the model
        # is retraining (RandomForest fit can take a moment with many samples).
        training_label = tk.Label(
            self.root,
            text="Training model…",
            font=_f(18, "bold"),
            fg=SUCCESS,
            bg=BG_ELEVATED,
        )
        training_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.root.update_idletasks()

        try:
            # Convert list of (63,) arrays to (N, 63) matrix
            features_array = np.array(features_list, dtype=np.float64)

            # Save to data/custom/<sign_name>/features.npy
            sign_dir = os.path.join(CUSTOM_DATA_DIR, sign_name)
            os.makedirs(sign_dir, exist_ok=True)
            npy_path = os.path.join(sign_dir, "features.npy")

            # If existing data, append to it
            if os.path.isfile(npy_path):
                try:
                    existing = np.load(npy_path)
                    features_array = np.vstack([existing, features_array])
                except Exception:
                    pass  # corrupted file — overwrite with new data

            np.save(npy_path, features_array)

            # Add to model and retrain
            self.model.add_training_data(sign_name, features_array)
            result = self.model.train()
            self.model.save_model()

            # Report results
            acc = result["accuracy"] if result else 0
            n = result["num_samples"] if result else 0
            training_label.destroy()
            messagebox.showinfo(
                "Training Complete",
                f"Sign '{sign_name}' saved!\n\n"
                f"Samples: {features_array.shape[0]}\n"
                f"Total training data: {n}\n"
                f"Model accuracy: {acc:.1%}",
            )
        except Exception as e:
            training_label.destroy()
            messagebox.showerror(
                "Save Error",
                f"Failed to save training data:\n{e}",
            )

        self.show_home()

    def _on_sequences_save(self, sign_name: str, sequences_list: list) -> None:
        """Save dynamic sequences to disk, retrain LSTM, and go home."""
        training_label = tk.Label(
            self.root,
            text="Training LSTM…",
            font=_f(18, "bold"),
            fg=ACCENT,
            bg=BG_ELEVATED,
        )
        training_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.root.update_idletasks()

        try:
            # Persist new sequences
            sign_seq_dir = os.path.join(SEQUENCE_DATA_DIR, sign_name)
            os.makedirs(sign_seq_dir, exist_ok=True)
            new_seqs = np.array(sequences_list, dtype=np.float32)
            seq_path = os.path.join(sign_seq_dir, "sequences.npy")
            if os.path.isfile(seq_path):
                try:
                    existing = np.load(seq_path)
                    new_seqs = np.concatenate([existing, new_seqs], axis=0)
                except Exception:
                    pass
            np.save(seq_path, new_seqs)

            # Collect all saved sequences for training
            all_sequences: dict = {}
            if os.path.isdir(SEQUENCE_DATA_DIR):
                for sname in os.listdir(SEQUENCE_DATA_DIR):
                    sdir = os.path.join(SEQUENCE_DATA_DIR, sname)
                    spath = os.path.join(sdir, "sequences.npy")
                    if os.path.isfile(spath):
                        try:
                            all_sequences[sname] = np.load(spath)
                        except Exception:
                            pass

            if not all_sequences:
                raise ValueError("No sequence data found to train on.")

            result = self.lstm_model.train(all_sequences)
            self.lstm_model.save(LSTM_MODEL_FILE, LSTM_LABELS_FILE)

            acc = result.get("accuracy", 0)
            n   = result.get("num_sequences", 0)
            training_label.destroy()
            messagebox.showinfo(
                "LSTM Training Complete",
                f"Dynamic sign '{sign_name}' saved!\n\n"
                f"Sequences used: {n}\n"
                f"Model accuracy: {acc:.1%}",
            )
        except Exception as e:
            training_label.destroy()
            messagebox.showerror(
                "LSTM Save Error",
                f"Failed to save dynamic sign data:\n{e}",
            )

        self.show_home()

    def _on_close(self):
        """Clean up resources before exiting."""
        # Warn if there's unsaved training data
        if self.training_screen.has_unsaved_data():
            if not messagebox.askyesno(
                "Unsaved Data",
                "You have unsaved training captures.\n"
                "Are you sure you want to quit?",
            ):
                return

        self.training_screen.cleanup()
        self.prediction_screen.cleanup()

        # Release shared resources
        self.camera.stop()
        self.detector.close()
        self.pose_detector.close()
        self.holistic.close()
        self.speaker.stop()

        self.root.destroy()

    def run(self):
        """Start the tkinter main loop."""
        self.root.mainloop()


def main():
    """Launch the sign language recognition application."""
    app = App()
    app.run()


if __name__ == "__main__":
    main()
