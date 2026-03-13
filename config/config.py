"""
config/config.py
----------------
Central configuration for Phase 1 & Phase 2.
All hyper-parameters, paths, and constants are defined here so that
every other module imports from a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Absolute path to the project root regardless of where the script is called
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "IMDB Dataset.csv")

# Processed artefacts (vocabulary, split indices, etc.) are saved here
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data-split hyper-parameters
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.80   # 80 % of the full dataset
VAL_RATIO:   float = 0.10   # 10 % of the full dataset
TEST_RATIO:  float = 0.10   # 10 % of the full dataset
RANDOM_STATE: int  = 42     # Fixed seed → reproducible splits

# ---------------------------------------------------------------------------
# Column names (as they appear in the CSV)
# ---------------------------------------------------------------------------
TEXT_COL:      str = "review"
LABEL_COL:     str = "sentiment"
LABEL_MAP: dict    = {"negative": 0, "positive": 1}

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
MAX_VOCAB_SIZE: int = 10_000   # Includes the 2 special tokens below

PAD_TOKEN: str = "<PAD>"       # Padding token
UNK_TOKEN: str = "<UNK>"       # Out-of-vocabulary token

PAD_IDX: int = 0               # Must stay at 0 for PyTorch padding_idx
UNK_IDX: int = 1

# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------
MAX_SEQ_LEN: int = 200         # Pad / truncate every review to this length

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 64
NUM_WORKERS: int = 0           # Set > 0 on Linux/macOS for faster loading

# ---------------------------------------------------------------------------
# Phase 2 – Model architecture (fixed across all architectures)
# ---------------------------------------------------------------------------
EMBED_DIM:   int = 128        # Embedding dimension shared by all models
N_LAYERS:    int = 2          # Number of stacked recurrent layers
OUTPUT_DIM:  int = 1          # Binary classification → single logit

# ---------------------------------------------------------------------------
# Phase 2 – Training
# ---------------------------------------------------------------------------
N_FOLDS:         int   = 3     # K-Fold cross-validation folds
N_EPOCHS_SEARCH: int   = 5     # Epochs per fold during hyperparameter search
N_EPOCHS_FINAL:  int   = 10    # Epochs for final retrain on full train set
GRAD_CLIP:       float = 1.0   # max_norm for gradient clipping

# ---------------------------------------------------------------------------
# Phase 2 – Hyperparameter grid (searched for every architecture)
# ---------------------------------------------------------------------------
PARAM_GRID: dict = {
    "hidden_dim": [128, 256, 512],
    "dropout":    [0.3, 0.5],
    "lr":         [1e-3, 1e-4],
    "optimizer":  ["adam", "rmsprop"],   # Adam vs RMSProp
}

# ---------------------------------------------------------------------------
# Phase 2 – Results / checkpoints directory
# ---------------------------------------------------------------------------
RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Phase 2 – Cached preprocessed data (avoids re-running spaCy each time)
# ---------------------------------------------------------------------------
CACHE_PATH: str = os.path.join(OUTPUTS_DIR, "phase1_cache.pkl")
