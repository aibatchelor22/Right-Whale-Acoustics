import random

import numpy as np
import librosa


# =========================================
# TIME SHIFT
# =========================================

def time_shift(
    audio,
    max_shift_sec,
    sr
):

    max_shift = int(
        max_shift_sec * sr
    )

    shift = random.randint(
        -max_shift,
        max_shift
    )

    shifted = np.roll(
        audio,
        shift
    )

    return shifted


# =========================================
# ADD NOISE
# =========================================

def add_noise(
    audio,
    noise_level=0.005
):

    noise = np.random.randn(
        len(audio)
    )

    augmented = (
        audio
        + noise_level * noise
    )

    return augmented.astype(
        np.float32
    )


# =========================================
# PITCH SHIFT
# =========================================

def pitch_shift(
    audio,
    sr,
    steps=1
):

    shifted = librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=steps
    )

    return shifted.astype(
        np.float32
    )


# =========================================
# TIME STRETCH
# =========================================

def time_stretch(
    audio,
    rate=1.05
):

    stretched = librosa.effects.time_stretch(
        audio,
        rate=rate
    )

    return stretched.astype(
        np.float32
    )


# =========================================
# SPECAUGMENT
# =========================================

def frequency_mask(
    spec,
    max_width=10
):

    spec = spec.copy()

    n_mels = spec.shape[0]

    width = random.randint(
        1,
        max_width
    )

    start = random.randint(
        0,
        n_mels - width
    )

    spec[
        start:start + width,
        :
    ] = 0

    return spec


def time_mask(
    spec,
    max_width=10
):

    spec = spec.copy()

    n_frames = spec.shape[1]

    width = random.randint(
        1,
        max_width
    )

    start = random.randint(
        0,
        n_frames - width
    )

    spec[
        :,
        start:start + width
    ] = 0

    return spec


# =========================================
# RANDOM AUGMENTATION PIPELINE
# =========================================

def augment_audio(
    audio,
    sr
):

    if random.random() < 0.5:

        audio = time_shift(
            audio,
            max_shift_sec=0.25,
            sr=sr
        )

    if random.random() < 0.3:

        audio = add_noise(
            audio,
            noise_level=0.003
        )

    return audio


def augment_spectrogram(
    spec
):

    if random.random() < 0.5:

        spec = frequency_mask(
            spec,
            max_width=8
        )

    if random.random() < 0.5:

        spec = time_mask(
            spec,
            max_width=8
        )

    return spec
