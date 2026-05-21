import random
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import librosa

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


# =========================================
# CONFIG
# =========================================

SR = 2000

WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(SR * WINDOW_SEC)

NEGATIVE_RATIO = 4

N_FFT = 1024
HOP_LENGTH = 64
N_MELS = 128

FMIN = 20
FMAX = 250

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================================
# PATHS
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = PROJECT_ROOT / "data/audio"

CSV_PATH = (
    PROJECT_ROOT
    / "data/right_whale_data/sbnms_200903_nopp6_ch10/detections/NEFSC_SBNMS_200903_NOPP6_CH10_upcall-detection-log.csv"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data/spec_cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================
# LOAD ANNOTATIONS
# =========================================

df = pd.read_csv(CSV_PATH)

df["Start_DateTime_ISO8601"] = pd.to_datetime(
    df["Start_DateTime_ISO8601"]
)

df["End_DateTime_ISO8601"] = pd.to_datetime(
    df["End_DateTime_ISO8601"]
)

print("Annotations:", len(df))

# =========================================
# AUDIO FILE TABLE
# =========================================

audio_files = sorted(
    AUDIO_DIR.glob("*.wav")
)

file_records = []

for wav_path in audio_files:

    parts = wav_path.stem.split("_")

    date_str = parts[2]
    time_str = parts[3]

    file_start = pd.to_datetime(
        date_str + time_str,
        format="%Y%m%d%H%M%S"
    ).tz_localize("Etc/GMT+5")

    file_end = (
        file_start
        + timedelta(minutes=15)
    )

    file_records.append({
        "path": wav_path,
        "start": file_start,
        "end": file_end,
    })

# =========================================
# FILE-LEVEL SPLIT
# =========================================

wav_paths = [
    rec["path"]
    for rec in file_records
]

train_files, temp_files = train_test_split(
    wav_paths,
    test_size=0.30,
    random_state=RANDOM_SEED
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.50,
    random_state=RANDOM_SEED
)

split_lookup = {}

for p in train_files:
    split_lookup[str(p)] = "train"

for p in val_files:
    split_lookup[str(p)] = "val"

for p in test_files:
    split_lookup[str(p)] = "test"

# =========================================
# MATCH POSITIVES
# =========================================

examples_by_split = {
    "train": [],
    "val": [],
    "test": []
}

annotations_by_file = {}

for _, row in tqdm(df.iterrows(), total=len(df)):

    ann_start = row["Start_DateTime_ISO8601"]
    ann_end = row["End_DateTime_ISO8601"]

    ann_mid = (
        ann_start
        + (ann_end - ann_start) / 2
    )

    for rec in file_records:

        if rec["start"] <= ann_mid < rec["end"]:

            rel_sec = (
                ann_mid - rec["start"]
            ).total_seconds()

            ex = {
                "wav_path": rec["path"],
                "center_sec": rel_sec,
                "label": 1
            }

            split = split_lookup[
                str(rec["path"])
            ]

            examples_by_split[
                split
            ].append(ex)

            key = str(rec["path"])

            if key not in annotations_by_file:
                annotations_by_file[key] = []

            annotations_by_file[key].append(
                rel_sec
            )

            break

# =========================================
# NEGATIVE SAMPLING
# =========================================

for rec in tqdm(file_records):

    wav_path = rec["path"]

    positives = annotations_by_file.get(
        str(wav_path),
        []
    )

    split = split_lookup[str(wav_path)]

    n_neg = max(
        1,
        len(positives) * NEGATIVE_RATIO
    )

    for _ in range(n_neg):

        for _attempt in range(100):

            candidate = random.uniform(
                WINDOW_SEC / 2,
                900 - WINDOW_SEC / 2
            )

            too_close = False

            for p in positives:

                if abs(candidate - p) < WINDOW_SEC:
                    too_close = True
                    break

            if not too_close:

                examples_by_split[
                    split
                ].append({
                    "wav_path": wav_path,
                    "center_sec": candidate,
                    "label": 0
                })

                break

# =========================================
# SPECTROGRAM
# =========================================

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

    spec_db = np.clip(spec_db, -80, 0)

    spec_db = (
        spec_db + 80
    ) / 80

    return spec_db.astype(np.float32)

# =========================================
# CACHE SPLIT
# =========================================

def cache_split(split_name):

    split_dir = CACHE_DIR / split_name

    split_dir.mkdir(exist_ok=True)

    metadata = []

    examples = examples_by_split[
        split_name
    ]

    for i, ex in enumerate(tqdm(examples)):

        y, sr = librosa.load(
            ex["wav_path"],
            sr=SR
        )

        center_sample = int(
            ex["center_sec"] * sr
        )

        start_sample = (
            center_sample
            - WINDOW_SAMPLES // 2
        )

        end_sample = (
            start_sample
            + WINDOW_SAMPLES
        )

        if start_sample < 0:
            continue

        if end_sample > len(y):
            continue

        clip = y[start_sample:end_sample]

        spec = make_spectrogram(clip)

        out_path = split_dir / f"{i}.npy"

        np.save(out_path, spec)

        metadata.append({
            "spec_path": out_path,
            "label": ex["label"]
        })

    meta_df = pd.DataFrame(metadata)

    meta_df.to_csv(
        CACHE_DIR / f"{split_name}.csv",
        index=False
    )

# =========================================
# RUN
# =========================================

cache_split("train")
cache_split("val")
cache_split("test")

print("DONE.")
