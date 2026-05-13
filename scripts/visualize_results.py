# visualization script — confusion matrix, per-class metrics, and training curves
#
# i originally only had the confusion matrix. added the per-class precision/recall/f1
# bars after i noticed sad had 0% recall and accuracy alone wasnt showing that at all.
# the bar chart made it obvious which classes were being ignored.
#
# added training curves after my supervisor asked which epoch the best checkpoint came from.
# the best_epoch marker on the plot makes that immediately obvious.
#
# the autolabel helper for bar annotations was needed because matplotlib doesnt add
# value labels automatically — had to look that up.
#
# needs model_outputs/training_history.json from lstm_train.py. if it doesnt exist
# the training curves are just skipped, everything else still works.
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from lstm_train import VideoLSTM, JsonVideoDataset, NUM_CLASSES, EMOTIONS, DEVICE, SEQUENCE_LENGTH, IMG_SIZE


def plot_confusion_matrix(y_true, y_predicted, classes, save_path="confusion_matrix.png"):
    """
    plots confusion matrix as a heatmap. rows = true label, cols = predicted.
    viridis colormap gives better contrast than the matplotlib default,
    makes it easier to spot where predictions cluster. seaborn handles the annotations.
    """
    cm = confusion_matrix(y_true, y_predicted)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})      # viridis has better contrast than default, easier to see which errors dominate

    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_class_metrics(y_true, y_predicted, classes, save_path="per_class_metrics.png"):
    """
    grouped bar chart of precision/recall/f1 per emotion class.
    this is what actually revealed that sad had 0% recall in my early experiments —
    overall accuracy looked fine but this plot made the problem impossible to miss.
    """
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_predicted, average=None, zero_division=0) # per-class, no averaging

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, p, width, label='Precision')
    rects2 = ax.bar(x, r, width, label='Recall')
    rects3 = ax.bar(x + width, f1, width, label='F1')

    ax.set_ylabel('Scores')
    ax.set_title('Per-class metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')


    # matplotlib doesnt annotate bars automatically so i had to add this
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

def plot_training_curves(history_path="model_outputs/training_history.json", save_dir="model_outputs"):
    """
    plots train/val loss and accuracy curves with a marker at the best epoch.
    added this after my supervisor asked which epoch the best checkpoint came from —
    the best_epoch marker makes it immediately obvious without having to dig through logs.
    """

    if not os.path.exists(history_path):
        print(f"Training_history.json not found at {history_path}")
        print("Run lstm_train.py first to generate training history.")
        return

    with open(history_path, "r") as f:
        h = json.load(f)

    epochs  = list(range(1, h["epochs_run"] + 1))
    best_ep = h["best_epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    ax1.plot(epochs, h["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs, h["val_loss"],   label="Val Loss",   color="#FF5722", linewidth=2)
    ax1.axvline(best_ep, color="green", linestyle="--", linewidth=1.5,
                label=f"Best epoch ({best_ep})")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss vs Epoch", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, h["train_acc"], label="Train Acc", color="#2196F3", linewidth=2)
    ax2.plot(epochs, h["val_acc"],   label="Val Acc",   color="#FF5722", linewidth=2)
    ax2.axvline(best_ep, color="green", linestyle="--", linewidth=1.5,
                label=f"Best epoch ({best_ep})")
    best_val = h["val_acc"][best_ep - 1]
    ax2.scatter([best_ep], [best_val], color="green", zorder=5, s=100,
                marker="*", label=f"Best val acc ({best_val:.1f}%)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs Epoch", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """
    loads the best checkpoint, runs predictions on the test set,
    then calls the three plot functions and saves everything to model_outputs/.
    """
    OUTPUT_DIR = "model_outputs"         # create output dir if this is the first run
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # best checkpoint — last epoch is usually slightly worse
    print(f">>> loading best model from models/emotion_lstm.pth...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_path = "data/test.json"
    if not os.path.exists(test_path):
        test_path = "25-26_CE903-SP_team02/ce903-emotion/data/test.json"

    if not os.path.exists(test_path):
        print(f"Error: Could not find test.json at {test_path}")
        return

    ds_te = JsonVideoDataset(test_path, sequence_length=SEQUENCE_LENGTH, transform=transform, phase="test")
    dl_te = DataLoader(ds_te, batch_size=4, shuffle=False)

    model = VideoLSTM(num_classes=NUM_CLASSES).to(DEVICE)
    model_path = "models/emotion_lstm.pth"

    if not os.path.exists(model_path):
        model_path = "25-26_CE903-SP_team02/ce903-emotion/models/emotion_lstm.pth"

    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(">>> running predictions...")
    y_pred = []
    y_true = []

    with torch.no_grad():
        for videos, labels in dl_te:
            videos, labels = videos.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            logits = model(videos)
            pred = logits.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    print(">>> generating plots...")

    plot_confusion_matrix(y_true, y_pred, classes=EMOTIONS, save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    plot_per_class_metrics(y_true, y_pred, classes=EMOTIONS, save_path=os.path.join(OUTPUT_DIR, "per_class_metrics.png"))

    plot_training_curves(
        history_path=os.path.join(OUTPUT_DIR, "training_history.json"),
        save_dir=OUTPUT_DIR
    )

    print(f"\nDone! All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
