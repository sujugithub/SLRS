# Sign Language Recognition App

A real-time desktop application for training and recognising custom hand sign gestures using your webcam. Built with Python, PyQt6, MediaPipe, and scikit-learn — no cloud, no internet required.

## Features

- **Dual-model recognition** — Random Forest for static signs, MLP temporal classifier for dynamic gestures
- **Holistic detection** — simultaneous hand (21 pts), pose (33 pts), and face (478 pts) landmark tracking via MediaPipe
- **Custom sign training** — record new signs in seconds; model retrains automatically
- **Sentence builder** — chains recognised signs into sentences with rule-based NLP correction
- **Text-to-speech** — non-blocking priority queue; uses macOS `say` or pyttsx3 as fallback
- **Pretrained signs** — `hello`, `peace`, `thumbs_up` included out of the box
- **Dark-theme PyQt6 UI** — glass-card design, smooth fade animations, threaded camera worker

## How It Works

### Static Signs (Random Forest)
1. MediaPipe detects up to 2 hands per frame (21 landmarks × 2 × 3 axes = 126 values)
2. Landmarks are normalised relative to the wrist and scaled to \[-1, 1\]
3. A `RandomForestClassifier` (100 estimators) predicts the sign with a confidence score
4. Predictions above 0.50 confidence are accepted

### Dynamic Gestures (MLP Temporal Classifier)
1. `HolisticDetector` captures a 171-value feature vector per frame (126 hand + 18 pose + 27 face)
2. A 30-frame rolling buffer (`SequenceCollector`) accumulates ~1 second of motion
3. Temporal statistics are extracted per feature dimension: **mean, std, displacement, velocity** → 732-D input
4. An `MLPClassifier` (layers: 256 → 128 → 64, Adam, early stopping) classifies the gesture
5. Predictions above 0.55 confidence are accepted

## Installation

### Prerequisites
- Python 3.9 or higher
- A webcam

### Setup

```bash
git clone <repo-url>
cd sign_language_app

pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Webcam capture and frame processing |
| `mediapipe` | Hand / pose / face landmark detection (Tasks API) |
| `scikit-learn` | RandomForest + MLPClassifier |
| `joblib` | Model persistence |
| `numpy` | Numerical operations |
| `pillow` | Image format conversion |
| `pyttsx3` | Cross-platform TTS fallback |
| `PyQt6` | GUI framework |
| `transformers` + `torch` | *(optional)* T5 grammar correction |

## Running the App

```bash
cd sign_language_app
python main.py
```

### Camera index

By default the app uses camera index `2`. Change it in `config.py` if needed:

```python
CAMERA_INDEX = 0  # set to your webcam index
```

## Training New Signs

### Static sign
1. Click **Train Signs** on the home screen
2. Enter a sign name (e.g. `thank_you`)
3. Hold your gesture in front of the camera
4. Click **Start Capturing** — frames are auto-collected every 300 ms
5. Click **Save & Train** — the model retrains in the background; accuracy is shown on completion

### Dynamic gesture
1. Switch to **Dynamic Mode** on the training screen
2. Enter a gesture name and click **Start Recording**
3. Perform the motion; the app captures 30-frame sequences automatically
4. Collect at least 5 sequences, then click **Train LSTM**

### Tips for better accuracy
- Capture 30–100 samples with slight variations in angle and position
- Use good, even lighting
- Keep your hand fully within the frame
- For dynamic gestures, perform the motion at a natural speed

## Generating Pretrained Data

```bash
python scripts/generate_pretrained.py
```

An interactive CLI walks you through capturing 20 samples each for 5 built-in signs (`peace`, `hello`, `thumbs_up`, `stop`, `point`). Press `s` to skip a sign or `q` to quit.

## Project Structure

```
sign_language_app/
├── main.py                       # App entry point, MainWindow, background workers
├── config.py                     # All configuration constants
├── requirements.txt
│
├── core/
│   ├── camera_handler.py         # OpenCV webcam wrapper
│   ├── camera_worker.py          # QThread: camera capture + live detection
│   ├── hand_detector.py          # MediaPipe hand landmarker (21 pts, dual-hand)
│   ├── pose_detector.py          # MediaPipe pose landmarker (33 pts)
│   ├── holistic_detector.py      # Unified hand + pose + face detector (171 D)
│   ├── feature_extractor.py      # Feature normalisation utilities
│   ├── model_trainer.py          # RandomForest static classifier
│   ├── lstm_trainer.py           # MLP temporal classifier
│   ├── sequence_collector.py     # 30-frame rolling buffer
│   ├── sentence_buffer.py        # Word queue (max 25 words, undo support)
│   ├── nlp_processor.py          # Rule-based NLP + optional T5 corrector
│   ├── tts_speaker.py            # Priority-queue TTS (non-blocking)
│   └── temporal_smoother.py      # Prediction smoothing helpers
│
├── gui/
│   ├── design.py                 # Design system: colours, QSS, reusable widgets
│   ├── home_screen.py            # Main menu with sign list and status indicator
│   ├── training_screen.py        # Static + dynamic capture interface
│   └── prediction_screen.py      # Live inference, sentence builder, TTS controls
│
├── models/
│   ├── hand_landmarker.task      # MediaPipe hand model (binary, pre-downloaded)
│   ├── pose_landmarker_lite.task # MediaPipe pose model
│   ├── face_landmarker.task      # MediaPipe face model
│   ├── sign_language_model.pkl   # Generated: RF classifier + labels
│   └── lstm_sign_model.keras     # Generated: MLP pipeline + labels JSON
│
├── data/
│   ├── pretrained/               # Built-in signs (hello, peace, thumbs_up)
│   ├── custom/                   # User-trained static sign features (.npy)
│   └── sequences/                # User-trained dynamic gesture sequences (.npy)
│
└── scripts/
    └── generate_pretrained.py    # Interactive CLI for capturing pretrained data
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `2` | Webcam device index |
| `CAMERA_WIDTH / HEIGHT` | `640 × 480` | Capture resolution |
| `MAX_NUM_HANDS` | `2` | Max hands detected per frame |
| `MIN_DETECTION_CONFIDENCE` | `0.7` | Hand detection threshold |
| `HOLISTIC_FEATURE_LENGTH` | `171` | 126 hand + 18 pose + 27 face values |
| `SEQ_LENGTH` | `30` | Frames per dynamic gesture window |
| `MIN_SEQUENCES_PER_SIGN` | `5` | Minimum sequences before MLP training |
| `WINDOW_WIDTH / HEIGHT` | `900 × 700` | PyQt6 window size |

## Architecture Overview

```
Webcam
  └─► CameraWorker (QThread)
        ├─► HandDetector      ─► 126-D features ─► RandomForest  ─► static sign
        └─► HolisticDetector  ─► 171-D features ─► SequenceCollector (30 frames)
                                                       └─► MLP Classifier
                                                                 │
                                                          SentenceBuffer
                                                                 │
                                                          RuleBasedNLP / T5
                                                                 │
                                                           TTSSpeaker
```

## License

This project is for educational purposes.
