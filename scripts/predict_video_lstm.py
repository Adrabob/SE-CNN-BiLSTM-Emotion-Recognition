# single / batch video inference — run from project root, not from scripts/ folder


# originally i used MediaPipe for face detection here because it seemed more modern
# and i thought better detection = better predictions. the model was outputting "fearful" for literally every video regardless of input. took me way too long to figure out
# that the issue was the face detector training used Haar Cascade via extract_faces.py
# but i was running MediaPipe at inference time, so the crops were completely different.
# switched to Haar Cascade with the same params and it started working.

#Also had a cv2 BGR vs RGB bug that was silently corrupting all the face crops.
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import argparse
from lstm_train import VideoLSTM, EMOTIONS, IMG_SIZE, SEQUENCE_LENGTH

# ground truth label mappings for auto-evaluation
RAVDESS_EMOTION = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fearful"}
CREMA_D_EMOTION = {"ANG": "angry", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# has to match extract_faces.py or crops will be wrong
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def parse_ground_truth(filename):
    """
    extracts ground truth emotion label from the video filename.
    supports RAVDESS and CREMA-D formats.
    returns "unknown" if the format is not recognized.
    """
    name = os.path.splitext(filename)[0]

    # RAVDESS: 7 dash-separated fields, emotion is at index 2
    parts = name.split("-")
    if len(parts) == 7:
        return RAVDESS_EMOTION.get(parts[2], "unknown")

    # CREMA-D: underscore-separated, emotion code is at index 2
    parts = name.split("_")
    if len(parts) >= 3:
        return CREMA_D_EMOTION.get(parts[2], "unknown")

    return "unknown"


def load_model(model_path):
    """loads model weights once so batch inference doesn't reload per video."""
    num_classes = len(EMOTIONS)
    model = VideoLSTM(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def predict_video(video_path, model):
    """
    reads all frames from the video, samples SEQUENCE_LENGTH evenly spaced ones,
    runs haar cascade on each to get the face crop, then passes the sequence through
    the model in one forward pass. zero tensors fill in for frames where no face was detected.
    returns dict with predicted emotion, confidence, and per-class probabilities.
    returns None if the video cannot be processed.
    """
    # no augmentation at inference, has to be identical to val_transform in training or you get a distribution shift
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     #Normalizes the pixel values of the input images using the mean and standard deviation values.
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open {os.path.basename(video_path)}")
        return None

    buf = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buf.append(frame)
    cap.release()

    if not buf:
        print(f"  Error: No frames extracted from {os.path.basename(video_path)}")
        return None

    # evenly spaced frames across the video
    indices = np.linspace(0, len(buf) - 1, SEQUENCE_LENGTH, dtype=int)

    face_seq = []
    n_det = 0

    for i in indices:
        frame = buf[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))   # exact same params as extract_faces.py — even small changes gave different crops and broke things

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])  # If there are multiple people, it picks the one closest to the camera(largest bounding box).

            fx, fy = max(0, fx), max(0, fy)
            fw = min(fw, frame.shape[1] - fx)
            fh = min(fh, frame.shape[0] - fy)

            face_crop = frame[fy:fy+fh, fx:fx+fw]
            if face_crop.size > 0:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
                face_seq.append(transform(face_resized))
                n_det += 1
            else:
                face_seq.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        else:
            face_seq.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))  # Blank frame if no detection

    if n_det == 0:
        print(f"  Warning: No faces detected in {os.path.basename(video_path)}")

    x = torch.stack(face_seq).unsqueeze(0).to(DEVICE, non_blocking=True)

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)     # softmax to get probabilities, model outputs raw logits not probs
        conf, pred = torch.max(probs, 1)

    idx = pred.item()
    predicted_emotion = EMOTIONS[idx]
    conf_pct = conf.item() * 100

    prob_dict = {emotion: probs[0][i].item() * 100 for i, emotion in enumerate(EMOTIONS)}

    return {"predicted": predicted_emotion, "confidence": conf_pct, "probabilities": prob_dict}        #Returns a dictionary containing the predicted emotion string


def print_single_result(video_path, result):
    """prints detailed output for single video mode (same format as before)."""
    if result is None:
        return
    print(f"\nvideo: {os.path.basename(video_path)}")
    print(f"Result: {result['predicted'].upper()} ({result['confidence']:.2f}%)")
    print("\nProbabilities:")
    for emotion in EMOTIONS:
        prob = result["probabilities"][emotion]
        marker = "<<" if emotion == result["predicted"] else ""
        print(f"  {emotion.ljust(10)}: {prob:.2f}% {marker}")


def batch_predict(folder_path, model_path):
    """
    runs inference on every .mp4 in the given folder, then prints a summary table
    with ground truth (parsed from filename), prediction, confidence, and correctness.
    """
    videos = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")])
    if not videos:
        print(f"No .mp4 files found in {folder_path}")
        return

    print(f"Loading model: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load model weights!\n{e}")
        print("Tip: Ensure the model was trained with the current lstm_train.py architecture.")
        return
    print(f"Device: {DEVICE}")
    print(f"Found {len(videos)} videos in {folder_path}\n")

    rows = []       #Stores the results
    for idx, fname in enumerate(videos, 1):
        vpath = os.path.join(folder_path, fname)
        gt = parse_ground_truth(fname)
        print(f"[{idx}/{len(videos)}] {fname} (gt: {gt}) ... ", end="", flush=True)

        result = predict_video(vpath, model)        #Runs inference using predict_video.
        if result is None:
            rows.append((fname, gt, "ERROR", 0.0, False))
            print("ERROR")
        else:
            correct = (result["predicted"] == gt) if gt != "unknown" else None
            rows.append((fname, gt, result["predicted"], result["confidence"], correct))
            mark = "?" if correct is None else ("OK" if correct else "MISS")
            print(f'{result["predicted"]} ({result["confidence"]:.1f}%) [{mark}]')

    # summary table
    col_w = [max(len(r[0]) for r in rows) + 2, 14, 14, 12, 9]
    headers = ["Video", "Ground Truth", "Predicted", "Confidence", "Result"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header_line = "|" + "|".join(h.center(w) for h, w in zip(headers, col_w)) + "|"

    print(f"\n{'=' * 60}")
    print("MANUAL TEST RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(sep)
    print(header_line)
    print(sep)

    n_correct = 0
    n_evaluated = 0

    for fname, gt, pred, conf, correct in rows:
        if correct is True:
            mark = "OK"
            n_correct += 1
            n_evaluated += 1
        elif correct is False:
            mark = "MISS"
            n_evaluated += 1
        else:
            mark = "?"

        row = "|" + "|".join([
            f" {fname}".ljust(col_w[0]),
            f" {gt}".ljust(col_w[1]),
            f" {pred}".ljust(col_w[2]),
            f" {conf:.2f}%".ljust(col_w[3]),
            f" {mark}".ljust(col_w[4]),
        ]) + "|"
        print(row)

    print(sep)

    # overall accuracy (only count videos where ground truth is known)
    if n_evaluated > 0:
        acc = n_correct / n_evaluated * 100
        acc_line = "|" + "|".join([
            " OVERALL".ljust(col_w[0]),
            f" {n_correct}/{n_evaluated}".ljust(col_w[1]),
            " ".ljust(col_w[2]),
            " ".ljust(col_w[3]),
            f" {acc:.1f}%".ljust(col_w[4]),
        ]) + "|"
        print(acc_line)
        print(sep)

    print(f"\nAccuracy: {n_correct}/{n_evaluated} ({acc:.1f}%)" if n_evaluated > 0 else "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Prediction from Video using LSTM")
    parser.add_argument("--video", type=str, default=None, help="Path to a single video file")      #If --video is provided, it loads the model, runs prediction, and prints the single result.
    parser.add_argument("--folder", type=str, default=None, help="Path to folder of test videos (default: data/manuel_test)")       #If --folder is provided (or defaults to data/manuel_test), it loads the model, runs batch prediction on all .mp4 files in the folder, and prints a summary table with results and overall accuracy.
    parser.add_argument("--model", type=str, default="models/emotion_lstm.pth", help="Path to the trained model file")      #Path to trainde model filen name defaults to models/emotion_lstm.pth.
    args = parser.parse_args()

    if args.video:
        # single video mode
        if not os.path.exists(args.video):
            print(f"Error: Video file not found at {args.video}")
        else:
            try:
                model = load_model(args.model)
                print(f"model loaded: {args.model}")
            except Exception as e:
                print(f"ERROR: Failed to load model weights!\n{e}")
                print("Tip: Ensure the model was trained with the current lstm_train.py architecture.")
                raise SystemExit(1)
            result = predict_video(args.video, model)
            print_single_result(args.video, result)
    else:
        # batch mode default folder is data/manuel_test
        folder = args.folder if args.folder else "data/manuel_test"
        if not os.path.isdir(folder):
            print(f"Error: Folder not found at {folder}")
        else:
            batch_predict(folder, args.model)
