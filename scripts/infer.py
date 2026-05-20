import argparse
from pathlib import Path

import numpy as np
import librosa

import torch

from whale_detector.model import WhaleCNN_TCN


# =========================================
# CONFIG
# =========================================

SR = 2000

WINDOW_SEC = 4.0

N_FFT = 1024
HOP_LENGTH = 64
N_MELS = 128

FMIN = 20
FMAX = 250

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =========================================
# ARGUMENTS
# =========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--wav",
    type=str,
    required=True
)

parser.add_argument(
    "--model",
    type=str,
    required=True
)

args = parser.parse_args()

# =========================================
# LOAD MODEL
# =========================================

model = WhaleCNN_TCN().to(DEVICE)

model.load_state_dict(
    torch.load(
        args.model,
        map_location=DEVICE
    )
)

model.eval()

# =========================================
# LOAD AUDIO
# =========================================

y, sr = librosa.load(
    args.wav,
    sr=SR
)

# =========================================
# WINDOW
# =========================================

window_samples = int(
    WINDOW_SEC * SR
)

clip = y[:window_samples]

# =========================================
# SPECTROGRAM
# =========================================

spec = librosa.feature.melspectrogram(
    y=clip,
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX
)

spec = librosa.power_to_db(
    spec,
    ref=np.max
)

spec = np.clip(spec, -80, 0)

spec = (spec + 80) / 80

spec = spec.astype(np.float32)

# =========================================
# TENSOR
# =========================================

x = torch.tensor(spec)

x = x.unsqueeze(0)

x = x.to(DEVICE)

# =========================================
# INFERENCE
# =========================================

with torch.no_grad():

    logits = model(x)

    prob = torch.sigmoid(
        logits
    ).item()

print()
print("Prediction Probability:")
print(prob)
