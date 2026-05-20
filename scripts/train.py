import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt

from whale_detector.dataset import WhaleSpecDataset
from whale_detector.model import WhaleCNN_TCN


# =========================================
# CONFIG
# =========================================

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
THRESHOLD = 0.40

DATA_DIR = Path("data/spec_cache")
RUNS_DIR = Path("runs")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================
# RUN DIRECTORY
# =========================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

run_dir = RUNS_DIR / f"run_{timestamp}"

run_dir.mkdir(parents=True, exist_ok=True)

print(f"Run directory: {run_dir}")

# =========================================
# DATASETS
# =========================================

train_ds = WhaleSpecDataset(
    DATA_DIR / "train.csv"
)

val_ds = WhaleSpecDataset(
    DATA_DIR / "val.csv"
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# =========================================
# MODEL
# =========================================

model = WhaleCNN_TCN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# =========================================
# TRAINING
# =========================================

train_losses = []
val_losses = []

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # -------------------------------------
    # TRAIN
    # -------------------------------------

    model.train()

    running_loss = 0.0

    for step, (x, y) in enumerate(train_loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if step % 20 == 0:
            print(
                f"Epoch {epoch+1} "
                f"Step {step} "
                f"Loss {loss.item():.4f}"
            )

    train_loss = (
        running_loss / len(train_loader)
    )

    train_losses.append(train_loss)

    # -------------------------------------
    # VALIDATION
    # -------------------------------------

    model.eval()

    val_running_loss = 0.0

    preds_all = []
    labels_all = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = criterion(logits, y)

            val_running_loss += loss.item()

            probs = torch.sigmoid(logits)

            preds = (
                probs >= THRESHOLD
            ).float()

            preds_all.extend(
                preds.cpu().numpy()
            )

            labels_all.extend(
                y.cpu().numpy()
            )

    val_loss = (
        val_running_loss / len(val_loader)
    )

    val_losses.append(val_loss)

    precision = precision_score(
        labels_all,
        preds_all
    )

    recall = recall_score(
        labels_all,
        preds_all
    )

    f1 = f1_score(
        labels_all,
        preds_all
    )

    print()
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")
    print()

    # -------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            run_dir / "best_model.pt"
        )

        print("Saved best model.")

# =========================================
# SAVE TRAINING CURVE
# =========================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Val Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    run_dir / "loss_curve.png",
    dpi=300
)

# =========================================
# SAVE METRICS
# =========================================

metrics = {
    "train_loss_final": train_losses[-1],
    "val_loss_final": val_losses[-1],
    "precision": precision,
    "recall": recall,
    "f1": f1,
}

with open(
    run_dir / "metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("Training complete.")
