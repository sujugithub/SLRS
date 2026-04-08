# SignBridge / SLRS

A real-time desktop app for recognizing and training sign-language gestures with a webcam. The current app is built with Python, PyQt6, MediaPipe Tasks, and scikit-learn, with all inference and training running locally.

## Current App Flow

The application launches directly into the live prediction screen. Additional flows are opened from the top-right menu drawer:

- Train new static or dynamic signs
- View and manage the saved vocabulary
- Retrain the static model from stored data
- Adjust runtime settings such as confidence threshold, TTS, camera index, and smoothing

## Features

- Real-time webcam inference with MediaPipe hand, pose, and face landmarks
- Static sign recognition using a Random Forest classifier
- Dynamic gesture recognition using temporal features plus a scikit-learn MLP classifier
- Sentence assembly with phrase matching and lightweight NLP cleanup
- Text-to-speech output with platform fallback support
- Persistent settings, phrase shortcuts, and training metadata stored on disk
- Desktop-first PyQt6 interface with a menu drawer, overlays, and live status indicators

## Recognition Pipeline

### Static signs
1. The hand detector extracts up to two hands of landmarks.
2. Static features are flattened to 126 values.
3. A `RandomForestClassifier` predicts the most likely sign.
4. The result is filtered by the configured confidence threshold.

### Dynamic gestures
1. The holistic detector produces a per-frame feature vector from hand, pose, and face landmarks.
2. A rolling sequence buffer keeps the last 30 frames.
3. Temporal descriptors are computed per feature: mean, standard deviation, displacement, and average absolute velocity.
4. Those engineered features are fed into an `MLPClassifier`.
5. The best prediction is emitted when it clears the confidence gate.

## Installation

### Prerequisites

- Python 3.9+
- A working webcam

### Setup

```bash
git clone https://github.com/sujugithub/SLRS.git
cd SLRS
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

By default the app uses camera index `2`. You can change that either in-app through Settings or by adjusting the defaults in `config.py`.

## Training Workflow

### Static sign training
1. Open the menu drawer.
2. Choose the training flow.
3. Enter a sign name.
4. Capture samples from the live camera feed.
5. Save and retrain the static model.

### Dynamic gesture training
1. Open the training flow and switch to dynamic mode.
2. Record gesture sequences.
3. Collect at least 5 sequences per sign.
4. Train the temporal model.

### Accuracy tips

- Record samples under consistent lighting.
- Vary angle and distance slightly so the model generalizes.
- Keep hands fully visible in frame.
- For motion signs, perform the gesture naturally and consistently.

## Data and Persistence

In development, data is read from and written to the repository folders. In packaged builds, writable app data is moved to the user profile:

- macOS: `~/Library/Application Support/SLRS`
- Windows: `%APPDATA%/SLRS`
- Linux: `~/.local/share/SLRS`

Important persisted files:

- `data/custom/` for static sign feature arrays
- `data/sequences/` for dynamic sign sequences
- `data/settings.json` for runtime settings
- `data/phrases.json` for phrase replacements
- `data/training_meta.json` for the latest retrain summary

## Project Structure

```text
SLRS/
├── main.py
├── config.py
├── requirements.txt
├── core/
│   ├── camera_handler.py
│   ├── camera_worker.py
│   ├── feature_extractor.py
│   ├── hand_detector.py
│   ├── holistic_detector.py
│   ├── lstm_trainer.py
│   ├── model_trainer.py
│   ├── nlp_processor.py
│   ├── phrase_store.py
│   ├── pose_detector.py
│   ├── sentence_buffer.py
│   ├── sequence_collector.py
│   ├── settings_store.py
│   ├── temporal_smoother.py
│   ├── training_meta.py
│   └── tts_speaker.py
├── gui/
│   ├── countdown_overlay.py
│   ├── design.py
│   ├── menu_overlay.py
│   ├── prediction_screen.py
│   ├── retrain_dialog.py
│   ├── settings_dialog.py
│   ├── train_dialog.py
│   ├── training_screen.py
│   └── view_dialog.py
├── models/
├── data/
└── scripts/
```

## Key Configuration

| Setting | Default | Description |
|---|---:|---|
| `CAMERA_INDEX` | `2` | Webcam device index |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640 x 480` | Capture resolution |
| `MAX_NUM_HANDS` | `2` | Maximum hands tracked |
| `MIN_DETECTION_CONFIDENCE` | `0.7` | MediaPipe detection threshold |
| `HOLISTIC_FEATURE_LENGTH` | `171` | Per-frame holistic feature width |
| `SEQ_LENGTH` | `30` | Frames per dynamic sequence |
| `MIN_SEQUENCES_PER_SIGN` | `5` | Minimum sequences before dynamic training |
| `WINDOW_WIDTH` / `WINDOW_HEIGHT` | `900 x 700` | Base window dimensions |

## Packaging

The repository includes platform build helpers and a PyInstaller spec:

```bash
./build.sh
```

On Windows, use `build.bat` instead.

## Notes

- Despite legacy filenames such as `lstm_trainer.py` and `lstm_sign_model.keras`, the current dynamic classifier is a scikit-learn MLP, not a TensorFlow LSTM.
- The static model is trained from the custom sign dataset stored on disk.

## License

This project is provided for educational purposes.
