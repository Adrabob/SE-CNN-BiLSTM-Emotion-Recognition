import os
import cv2
from tqdm import tqdm

# Baseline face detector (fast but flawed; sufficient for demo)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def iter_videos(root_dir: str):
    # root_dir: data/raw_videos/<label>/*.mp4
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                yield label, os.path.join(label_dir, fname)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def extract_faces_from_video(video_path: str, out_dir: str, stride: int = 5, min_size: int = 64):
    """
    stride=5 -> takes a sample every 5 frames (adjusted according to fps)
    min_size -> discards very small faces
    """
    ensure_dir(out_dir)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size),
        )

        # Choose the largest face (assuming a single person).
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            face = frame[y:y+h, x:x+w]
            # Standard size for the model input
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, face)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved

def main():
    raw_root = os.path.join("data", "raw_videos")
    out_root = os.path.join("data", "processed_faces")
    ensure_dir(out_root)

    videos = list(iter_videos(raw_root))
    if not videos:
        print("No videos found. Put videos under: data/raw_videos/<label>/*.mp4")
        return

    for label, vpath in tqdm(videos, desc="Processing videos"):
        vname = os.path.splitext(os.path.basename(vpath))[0]
        out_dir = os.path.join(out_root, label, vname)
        ensure_dir(out_dir)
        saved = extract_faces_from_video(vpath, out_dir, stride=5)
        # If it's empty, log it down.
        if saved == 0:
            print(f"[WARN] No faces saved for {vpath}")

    print("Done. Faces saved under data/processed_faces/")

if __name__ == "__main__":
    main()
