import numpy as np
import librosa

import torch

from whale_detector.spectrograms import (
    make_spectrogram
)

from whale_detector.config import (
    SR,
    WINDOW_SEC,
)


def predict_clip(
    model,
    wav_path,
    device
):

    y, sr = librosa.load(
        wav_path,
        sr=SR
    )

    n_samples = int(
        WINDOW_SEC * SR
    )

    clip = y[:n_samples]

    spec = make_spectrogram(
        clip
    )

    x = torch.tensor(
        spec,
        dtype=torch.float32
    )

    x = x.unsqueeze(0)

    x = x.to(device)

    model.eval()

    with torch.no_grad():

        logits = model(x)

        prob = torch.sigmoid(
            logits
        ).item()

    return prob
