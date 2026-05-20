import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

# =========================================
# CONFIG
# =========================================

SR = 2000

HOP_LENGTH = 64

FMIN = 20
FMAX = 500

CACHE_DIR = "data/spec_cache"

# =========================================
# LOAD CSV
# =========================================

train_df = pd.read_csv(
    f"{CACHE_DIR}/train.csv"
)

# =========================================
# RANDOM SAMPLES
# =========================================

pos_row = train_df[
    train_df["label"] == 1
].sample(1).iloc[0]

neg_row = train_df[
    train_df["label"] == 0
].sample(1).iloc[0]

pos_spec = np.load(
    pos_row["spec_path"]
)

neg_spec = np.load(
    neg_row["spec_path"]
)

# =========================================
# PLOT
# =========================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(18, 6),
    constrained_layout=True
)

# -----------------------------------------
# POSITIVE
# -----------------------------------------

img1 = librosa.display.specshow(
    pos_spec,
    sr=SR,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="hz",
    fmin=FMIN,
    fmax=FMAX,
    cmap="winter",
    ax=axs[0]
)

axs[0].set_title(
    "Positive Example"
)

# -----------------------------------------
# NEGATIVE
# -----------------------------------------

img2 = librosa.display.specshow(
    neg_spec,
    sr=SR,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="hz",
    fmin=FMIN,
    fmax=FMAX,
    cmap="winter",
    ax=axs[1]
)

axs[1].set_title(
    "Negative Example"
)

# =========================================
# COLORBAR
# =========================================

cbar = fig.colorbar(
    img1,
    ax=axs,
    shrink=0.85,
    pad=0.02
)

cbar.set_label(
    "Normalized dB"
)

# =========================================
# SAVE
# =========================================

plt.savefig(
    "sample_spectrograms.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
