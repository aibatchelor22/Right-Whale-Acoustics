from pathlib import Path

# =========================================
# PROJECT PATHS
# =========================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATA_DIR = PROJECT_ROOT / "data"

AUDIO_DIR = DATA_DIR / "audio"

DETECTION_DIR = DATA_DIR / "detections"

CACHE_DIR = DATA_DIR / "spec_cache"

RUNS_DIR = PROJECT_ROOT / "runs"

# =========================================
# AUDIO CONFIG
# =========================================

SR = 2000

WINDOW_SEC = 4.0

WINDOW_SAMPLES = int(
    SR * WINDOW_SEC
)

# =========================================
# SPECTROGRAM CONFIG
# =========================================

N_FFT = 1024

HOP_LENGTH = 64

N_MELS = 128

FMIN = 20

FMAX = 250

# =========================================
# TRAINING CONFIG
# =========================================

BATCH_SIZE = 64

LR = 1e-3

EPOCHS = 20

THRESHOLD = 0.40
