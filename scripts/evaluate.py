import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from whale_detector.dataset import WhaleSpecDataset
from whale_detector.model import WhaleCNN_TCN


# =========================================
# ARGUMENTS
# =========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "run_dir",
    type=str,
    help="Path to run directory"
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.40
)

args = parser.parse_args()

RUN_DIR = Path(args.run_dir)

THRESHOLD = args.threshold

# =========================================
# PATHS
# =========================================

DATA_DIR = Path("data/spec_cache")

MODEL_PATH = RUN_DIR / "best_model.pt"

OUTPUT_DIR = RUN_DIR / "evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================
# DEVICE
# =========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("Using device:", DEVICE)

# =========================================
# DATA
# =========================================

test_ds = WhaleSpecDataset(
    DATA_DIR / "test.csv"
)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False
)

# =========================================
# MODEL
# =========================================

model = WhaleCNN_TCN().to(DEVICE)

print()
print("Loading model:")
print(MODEL_PATH)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# =========================================
# INFERENCE
# =========================================

all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

        logits = model(x)

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()

        preds = (
            probs >= THRESHOLD
        ).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =========================================
# METRICS
# =========================================

precision = precision_score(
    all_labels,
    all_preds
)

recall = recall_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds
)

roc_auc = roc_auc_score(
    all_labels,
    all_probs
)

pr_auc = average_precision_score(
    all_labels,
    all_probs
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

report = classification_report(
    all_labels,
    all_preds
)

# =========================================
# PRINT RESULTS
# =========================================

print()
print("===== EVALUATION =====")
print()

print(f"Threshold: {THRESHOLD:.2f}")
print()

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print()
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print()
print("Confusion Matrix")
print(cm)

print()
print("Classification Report")
print(report)

# =========================================
# SAVE METRICS
# =========================================

metrics = {
    "threshold": float(THRESHOLD),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# =========================================
# SAVE CLASSIFICATION REPORT
# =========================================

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as f:

    f.write(report)

# =========================================
# ROC CURVE
# =========================================

fpr, tpr, _ = roc_curve(
    all_labels,
    all_probs
)

plt.figure(figsize=(6, 6))

plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=300
)

plt.close()

# =========================================
# PR CURVE
# =========================================

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        all_labels,
        all_probs
    )
)

plt.figure(figsize=(6, 6))

plt.plot(
    recall_curve,
    precision_curve
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pr_curve.png",
    dpi=300
)

plt.close()

print()
print("Saved evaluation outputs to:")
print(OUTPUT_DIR)
