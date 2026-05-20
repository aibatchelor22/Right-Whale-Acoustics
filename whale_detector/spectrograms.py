import numpy as np
import librosa

from whale_detector.config import (
    SR,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
)


def make_spectrogram(audio):

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    spec_db = librosa.power_to_db(
        spec,
        ref=np.max
    )

    spec_db = np.clip(
        spec_db,
        -80,
        0
    )

    spec_db = (
        spec_db + 80
    ) / 80

    return spec_db.astype(np.float32)
