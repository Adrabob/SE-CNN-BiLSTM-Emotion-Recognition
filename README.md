# SE-CNN + BiLSTM — Video Facial Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.54%25-brightgreen)

A deep learning system for **video-based facial emotion recognition** across five emotion classes. The model combines a Squeeze-Excitation CNN backbone with a Bidirectional LSTM and learned temporal attention to classify the emotional state of a person from a short video clip.

Trained and evaluated on the [RAVDESS](https://zenodo.org/record/1188976) dataset (24 actors, speech + song modalities).

---

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **96.54%** |
| Best Val Accuracy | 72.22% (epoch 21) |
| Epochs Trained | 30 |

| Emotion | F1 Score |
|---|---|
| Neutral | 0.96 (recall) |
| Happy | 0.75 |
| Angry | 0.81 |
| Fearful | 0.61 |
| Sad | 0.49 |

Evaluation outputs (confusion matrix, per-class metrics, training curves) are saved to `model_outputs/`.

---

## Architecture

```
Input: (batch, seq=10, 3, 112, 112)
         │
         ▼
 ┌─────────────────────────────────┐
 │  CNNEncoder  (per frame)        │
 │  ConvBlock × 5                  │
 │  [3 → 32 → 64 → 128 → 256 → 512]│
 │  Each block:                    │
 │    Conv2d → BatchNorm → GELU    │
 │    → SEBlock → MaxPool2d        │
 │  AdaptiveAvgPool2d → Flatten    │
 │  Dropout(0.3)                   │
 │  Output: 512-dim per frame      │
 └──────────────┬──────────────────┘
                │
                ▼
 ┌─────────────────────────────────┐
 │  BiLSTM                         │
 │  2 layers, hidden=256           │
 │  Bidirectional → 512-dim/step   │
 └──────────────┬──────────────────┘
                │
                ▼
 ┌─────────────────────────────────┐
 │  Temporal Attention             │
 │  Linear(512→1) + Softmax        │
 │  Weighted sum over timesteps    │
 └──────────────┬──────────────────┘
                │
                ▼
 ┌─────────────────────────────────┐
 │  Classifier                     │
 │  Dropout(0.5) + Linear(512→5)   │
 └─────────────────────────────────┘
```

**SEBlock** (Squeeze-Excitation) applies channel-wise attention inside each convolutional block, letting the network dynamically recalibrate feature map importance. The **BiLSTM** captures how emotion evolves over time, and the **attention layer** learns which frames in the sequence carry the strongest emotional signal.

---

## Tech Stack

| Component | Library |
|---|---|
| Model & Training | PyTorch 2.x, torchvision |
| Face Detection | OpenCV (Haar Cascade) |
| Data Augmentation | torchvision.transforms |
| Metrics & Evaluation | scikit-learn |
| Visualisation | matplotlib, seaborn |
| Image I/O | Pillow |
| Progress Bars | tqdm |

---

## Project Structure

```
SE-CNN_BILSTM/
├── scripts/
│   ├── lstm_train.py           # Model definition + full training loop
│   ├── predict_video_lstm.py   # Single-video and batch inference
│   ├── realtime_demo_lstm.py   # Live webcam demo
│   ├── extract_faces.py        # Haar Cascade face extraction from videos
│   ├── make_split.py           # Actor-level train/val/test split
│   ├── prepare_ravdess.py      # Organise raw RAVDESS files by emotion label
│   ├── visualize_results.py    # Generate evaluation plots
│   └── requirements.txt
├── data/
│   ├── ravdess_raw/            # Raw RAVDESS actor folders (not tracked)
│   ├── raw_videos/<label>/     # Videos sorted by emotion
│   ├── processed_faces/        # 224×224 face crops (not tracked)
│   ├── train.json              # 55,573 frame paths + labels
│   ├── val.json                # 14,573 frame paths + labels
│   ├── test.json               # 12,248 frame paths + labels
│   └── manuel_test/            # Manual test videos
├── models/
│   └── emotion_lstm.pth        # Best model checkpoint (saved at peak val accuracy)
└── model_outputs/
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    ├── training_curves.png
    └── training_history.json
```

---

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (falls back to CPU automatically)

### Install dependencies

```bash
pip install -r scripts/requirements.txt
```

For GPU support (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/requirements.txt
```

---

## Usage

All commands are run from the **project root** directory.

### 1. Prepare the dataset

Download [RAVDESS](https://zenodo.org/record/1188976) and place the actor folders under `data/ravdess_raw/`. Then:

```bash
# Sort raw files into emotion-labelled folders
python scripts/prepare_ravdess.py --src data/ravdess_raw --dst data/raw_videos

# Extract face crops (Haar Cascade, every 5th frame, 224×224)
python scripts/extract_faces.py

# Build actor-level train/val/test split (70/20/10, seed=42)
python scripts/make_split.py
```

### 2. Train

```bash
python scripts/lstm_train.py
```

Best weights are saved to `models/emotion_lstm.pth` whenever validation accuracy improves. Training history is written to `model_outputs/training_history.json`.

### 3. Evaluate

```bash
python scripts/visualize_results.py
```

Generates `confusion_matrix.png`, `per_class_metrics.png`, and `training_curves.png` in `model_outputs/`.

### 4. Inference

```bash
# Single video
python scripts/predict_video_lstm.py --video path/to/video.mp4

# Batch — prints a results table with ground truth parsed from RAVDESS filenames
python scripts/predict_video_lstm.py --folder data/manuel_test

# Live webcam demo (press Q to quit)
python scripts/realtime_demo_lstm.py
```

---

## Key Design Decisions

**Actor-level data split** — Random video-level splitting allowed the model to memorise individual actor faces, inflating accuracy. Enforcing that every frame from a given actor stays in one partition only forced the model to generalise to unseen identities.

**Consistent face detector** — Training uses Haar Cascade to extract face crops (`extract_faces.py`). An early inference version used MediaPipe BlazeFace, which produces geometrically different crop regions. The distribution mismatch caused the model to predict "fearful" for virtually every video. All scripts now use identical Haar Cascade parameters (`scaleFactor=1.1, minNeighbors=5, minSize=(64,64)`).

**Dual class-imbalance correction** — RAVDESS is heavily skewed toward neutral. `WeightedRandomSampler` alone was insufficient; weighted `CrossEntropyLoss` alone was insufficient. Both are required together to prevent neutral domination.

**CosineAnnealingLR over ReduceLROnPlateau** — `ReduceLROnPlateau` aggressively halved the learning rate until it collapsed to ~6e-6 and training stalled. `CosineAnnealingLR` provides smooth, predictable decay with no manual tuning.

**Temporal attention over last hidden state** — Using `lstm_out[:, -1, :]` (final hidden state) underperformed. Learned attention weights over all timesteps improved results, especially for emotions like sadness that build gradually across frames.

**5-layer CNN** — A 4-layer backbone plateaued at ~43% accuracy. The additional 512-channel convolutional block provided the representational capacity needed to separate subtle facial expressions.

---

## License

MIT
