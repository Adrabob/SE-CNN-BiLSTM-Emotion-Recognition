# real-time webcam emotion recognition using a sliding window buffer
#
# the model needs SEQUENCE_LENGTH frames to make a prediction, so i cant just classify
# one frame at a time. the fix is a sliding window: keep the last N frames in a deque
# and run the model every time a new frame comes in (once the buffer is full).
#
# i tried using a plain list with pop(0) first but deque with maxlen is cleaner and
# automatically drops old frames so i dont have to manage it manually.
#
# the mirror flip (cv2.flip) is important for usability — without it the webcam feed
# is mirrored which makes it feel really unnatural when you move. found this out the
# first time i tested it and it was immediately annoying.
#
# predictions show "Waiting..." until the buffer fills up for the first time.
# press q to quit.

import torch
import torchvision.transforms as transforms
import cv2
from collections import deque

from lstm_train import VideoLSTM, EMOTIONS, IMG_SIZE, SEQUENCE_LENGTH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Haar Cascade — same detector used during training (extract_faces.py).
# Switched back from MediaPipe after it gave different crop regions and
# the model started misclassifying almost everything as "fearful".
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def realtime_predict(model_path):
    """
    loads the model, opens the webcam, and runs inference on a rolling window of frames.
    the first SEQUENCE_LENGTH frames just fill the buffer without making a prediction.
    after that every new frame triggers a forward pass and updates the displayed label.
    """
    print(f"device: {DEVICE}")

    num_classes = len(EMOTIONS)
    model = VideoLSTM(num_classes=num_classes).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f"model loaded: {model_path}")
    except Exception as e:
        print(f"ERROR: Failed to load model weights!\n{e}")
        return

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    buf = deque(maxlen=SEQUENCE_LENGTH)  # deque auto-drops oldest frame when full, tried a plain list first but this is cleaner

    cur_emo = "Waiting..."
    cur_conf = 0.0

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flipping made it way less annoying to use
        img_h, img_w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
        )

        ft = None
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])

            fx, fy = max(0, fx), max(0, fy)
            fw = min(fw, img_w - fx)
            fh = min(fh, img_h - fy)

            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

            if fw > 0 and fh > 0:
                face_img = frame[fy:fy+fh, fx:fx+fw]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
                ft = transform(face_resized)

        if ft is None:
            ft = torch.zeros(3, IMG_SIZE, IMG_SIZE)  # zero tensor placeholder — model still runs, just with less info for that step

        buf.append(ft)

        # Need full buffer first. Otherwise we get a lot of random predictions at the start.
        if len(buf) == SEQUENCE_LENGTH:
            x = torch.stack(list(buf)).unsqueeze(0).to(DEVICE, non_blocking=True)

            with torch.no_grad():
                logits = model(x)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)

                cur_emo = EMOTIONS[pred.item()]
                cur_conf = conf.item() * 100

        text = f"{cur_emo.upper()} ({cur_conf:.1f}%)"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Real-Time Emotion LSTM', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_predict("models/emotion_lstm.pth")
