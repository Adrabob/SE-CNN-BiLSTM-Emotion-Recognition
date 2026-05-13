# SE-CNN + BiLSTM for facial emotion recognition — CE903 group project
#
#I originally started with a basic 4-layer CNN and accuracy just sat at ~43%. Added a 5th conv layer and SE blocks (squeeze-excitation, basically channel attention) and it finally started improving. also switched from ReLU to GELU which helped a bit.
#
# Biggest mistake i made was doing a random video-level split instead of actor-level. Turns out the model was just memorising individual faces, not learning emotions. switched to actor-disjoint split and accuracy dropped to 64%. depressing but at least its honest.

#For Evaluation Metrics, neutral was dominating and sad/fearful had literally 0% recall for a while. needed BOTH the weighted sampler AND weighted loss
# together to fix it. tried each one separately first, neither was enough on its own.


# chose LSTM over a transformer because ViT with 10 frames didnt fit in my gpu memory.
# BiLSTM + temporal attention worked pretty well anyway.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import os
import json
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from collections import defaultdict, Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")       #Utilises cuda if a GPU is available, otherwise defaults to cpu
BATCH_SIZE = 32 
EPOCHS = 30
BACKBONE_LR = 1e-4 # Cnn encoder learning rate. 
HEAD_LR = 3e-4 # lstm and classifier learning rate, higher than backbone lr
IMG_SIZE = 112
SEQUENCE_LENGTH = 10

# Defines the 5 emotion categories and maps string labels to integers.
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]
LABEL_MAP = {emotion: i for i, emotion in enumerate(EMOTIONS)}
NUM_CLASSES = len(EMOTIONS)


class JsonVideoDataset(Dataset):
    def __init__(self, json_path, sequence_length=16, transform=None, phase="train"):
        """
        loads video frame paths from a json file and groups them by video id.
        the json path resolution was annoying to get right — the paths in the json
        can be relative to different roots depending on where you run the script from,
        so __getitem__ tries a few variants. not pretty but it works.
        """
        self.phase = phase
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            self.items = json.load(f)

        self.sequence_length = sequence_length
        self.transform = transform

        self.video_groups = defaultdict(list)
        for item in self.items:
            if item['label'] in LABEL_MAP:      # Skip any labels not in our defined set
                self.video_groups[item['video']].append(item)

        self.video_ids = list(self.video_groups.keys())

        if phase == "train":
            labels = [self.video_groups[vid][0]['label'] for vid in self.video_ids]
            print(f"[{phase.upper()}] Class Distribution: {Counter(labels)}")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        returns a (sequence_length, 3, H, W) tensor and integer label.
        if the video has fewer frames than sequence_length, zero tensors pad the end.
        tried repeating frames instead of zero-padding but accuracy was basically the same.
        """
        v_id = self.video_ids[idx]
        frames_info = sorted(self.video_groups[v_id], key=lambda x: x['path'])

        label_str = frames_info[0]['label']
        label = LABEL_MAP[label_str]

        total_available = len(frames_info)
        
        #If the video contains more frames than SEQUENCE_LENGTH, 
        if total_available >= self.sequence_length:
            indices = np.linspace(0, total_available - 1, self.sequence_length, dtype=int)          #it samples frames evenly across the sequence using np.linspace
        else:
            indices = range(total_available)


        selected_frames = []
        for i in indices:
            rel_path = frames_info[i]['path'].replace("\\", "/")

            # Try a couple of path variants
            # rel_path_old = frames_info[i]['path']  # old version before path fix
            if os.path.exists(rel_path):
                full_path = rel_path

            elif os.path.exists(os.path.join("data", rel_path)):
                full_path = os.path.join("data", rel_path)

            else:
                full_path = os.path.join("25-26_CE903-SP_team02/ce903-emotion", rel_path)

            try:
                #Opens images using PIL, converts them to RGB, resizes to 112x112, and applies the specified transform.
                with Image.open(full_path) as img:
                    img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                    if self.transform:
                        tens = self.transform(img)
                        selected_frames.append(tens)

            except Exception as e:
                selected_frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

        while len(selected_frames) < self.sequence_length:
            selected_frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))


        return torch.stack(selected_frames), label


class SEBlock(nn.Module):
    """
    squeeze-excitation block. recalibrates channel weights based on global context.
    it lets the network learn which feature maps matter more.
    """


    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)      #This function reduces spatial dimensions (H, W) to 1x1 
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),     #Compresses channels by a factor of 16 (reduction=16)
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()    # Between 0 and 1
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.excitation(self.squeeze(x)).view(b, c, 1, 1)
        return x * scale


# tried relu first but switched to gelu after seeing better convergence, especially in later layers. not a huge difference but kept it.
class ConvBlock(nn.Module):         #actually this is a standard convolutional building block.
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return self.pool(x)


# went from 4 layers to 5 after accuracy got stuck around 43%. needed the extra capacity.
# 512 channels in the last block gives the cnn enough room to separate subtle expressions.
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.blocks = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
        )
        self.dropout = nn.Dropout(0.3)      #applies dropout to mitigate overfitting before returning a 1D feature vector per frame.

    def forward(self, x):
        return self.dropout(self.blocks(x))



# attention over time steps. tried using just the last hidden state first but it was worse.
# emotion signal is spread across frames so averaging with learned weights makes more sense.
class VideoLSTM(nn.Module):
    """
    full model: CNN per frame - BiLSTM over sequence - attention pooling - classify.
    i initially just used lstm_out[:, -1, :] (last hidden state) and it underperformed.
    switched to learned attention weights over all time steps after reading about it in a paper,
    got a noticeable improvement especially for emotions that build up gradually like sad.
    """

    def __init__(self, num_classes):
        super(VideoLSTM, self).__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = nn.Linear(512, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)

        features = self.encoder(x)
        # print(features.shape)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)

        logits = self.classifier(context)
        return logits


def train():
    """
    full training loop: data loading, class weighting, training, validation, early stopping.
    the two things that made the biggest difference were:
    1) using BOTH weighted sampler AND weighted loss (just one wasnt enough)
    2) switching from ReduceLROnPlateau to CosineAnnealingLR
    """
    print(f">>> device: {DEVICE}")
    print(f">>> classes: {EMOTIONS}")

    torch.backends.cudnn.benchmark = True

    os.makedirs("models", exist_ok=True)
    os.makedirs("model_outputs", exist_ok=True)

    train_loss_hist = []

    train_acc_hist = []

    val_loss_hist = []

    val_acc_hist = []
    best_epoch = 0

    train_transform = transforms.Compose([
        
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         # imagenet stats, backbone was pretrained on it so using different normalization really hurt accuracy
    ])

    val_transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = "data/train.json"
    val_path = "data/val.json"
    test_path = "data/test.json"

    if not os.path.exists(train_path):
        prefix = "25-26_CE903-SP_team02/ce903-emotion/"
        train_path = prefix + "data/train.json"
        val_path = prefix + "data/val.json"
        test_path = prefix + "data/test.json"

    print(f">>> loading data from: {train_path}")

    ds_tr = JsonVideoDataset(train_path, sequence_length=SEQUENCE_LENGTH, transform=train_transform, phase="train")
    ds_va = JsonVideoDataset(val_path, sequence_length=SEQUENCE_LENGTH, transform=val_transform, phase="val")
    ds_te = JsonVideoDataset(test_path, sequence_length=SEQUENCE_LENGTH, transform=val_transform, phase="test")

    labels_list = [LABEL_MAP[ds_tr.video_groups[vid][0]['label']] for vid in ds_tr.video_ids]
    cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=labels_list)
    class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)       # neutral was getting ignored without this
    print(f">>> class weights: { {EMOTIONS[i]: round(cw[i], 3) for i in range(NUM_CLASSES)} }")

    # both sampler and loss weights together. tried each one on its own first, neither was enough to fix neutral dominating
    sample_weights = [cw[label] for label in labels_list]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    num_workers = 4 if os.name != 'nt' else 2  # windos caps this differently, 0 on nt causes issues sometimes
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
                       num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = VideoLSTM(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # label_smoothin 0.1, without it the model got overconfident on easy classes

    opt = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': BACKBONE_LR},
        {'params': model.lstm.parameters(), 'lr': HEAD_LR},
        {'params': model.attention.parameters(), 'lr': HEAD_LR},
        {'params': model.classifier.parameters(), 'lr': HEAD_LR},
    ], weight_decay=1e-4)

    # cosine annealing. ReduceLROnPlateau kept collapsing the lr too aggressively
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    
    EARLY_STOP_PATIENCE = 10        # Early stoppingif no improvement in validation accuracy.
    best_val_acc = 0.0
    stale = 0  

    print(">>> training start")
    for ep in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(dl_tr, desc=f"Epoch {ep+1}/{EPOCHS}")
        for videos, y in pbar:
            videos, y = videos.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(videos)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            curr_loss = loss.item()
            train_loss += curr_loss
            pred = logits.argmax(dim=1)
            train_total += y.size(0)
            train_correct += (pred == y).sum().item()

            pbar.set_postfix(loss=curr_loss)

        train_acc = (train_correct / train_total) * 100

        model.eval()
        val_correct = 0
        val_total = 0
        vloss = 0.0
        with torch.no_grad():
            for videos, y in dl_va:
                videos, y = videos.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                logits = model(videos)
                # print(f"batch val loss: {criterion(logits, y).item()}")
                vloss += criterion(logits, y).item()
                pred = logits.argmax(dim=1)
                val_total += y.size(0)
                val_correct += (pred == y).sum().item()

        avg_val_loss = vloss / len(dl_va)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        train_loss_hist.append(train_loss / len(dl_tr))
        train_acc_hist.append(train_acc)
        val_loss_hist.append(avg_val_loss)
        val_acc_hist.append(val_acc)
        print(f"epoch {ep+1}: train={train_acc:.1f}%  val={val_acc:.1f}%")

        sch.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep + 1
            stale = 0
            torch.save(model.state_dict(), "models/emotion_lstm.pth")
            print(f">>> saved best model ({val_acc:.2f}%)")
        else:
            stale += 1

            if stale >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {ep+1}, no improvement for {EARLY_STOP_PATIENCE} epochs")
                break

    history = {}
    history["train_loss"] = train_loss_hist
    history["train_acc"] = train_acc_hist
    history["val_loss"] = val_loss_hist
    history["val_acc"] = val_acc_hist
    history["best_epoch"] = best_epoch
    history["epochs_run"] = len(train_loss_hist)
    with open("model_outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("saved: model_outputs/training_history.json")

    print("\n>>> test evaluation:")
    if os.path.exists("models/emotion_lstm.pth"):
        model.load_state_dict(torch.load("models/emotion_lstm.pth", weights_only=True))

    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for videos, y in dl_te:
            videos, y = videos.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits = model(videos)
            pred = logits.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    unique_labels = sorted(list(set(y_true)))
    target_names = [EMOTIONS[i] for i in unique_labels]

    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    train()
