"""
phase1_preprocessing.py
=======================
Entry point for Phase 1 – Data Acquisition & Preprocessing.

Execution order
---------------
1.  Load the raw IMDb CSV.
2.  Stratified train / val / test split  (80 / 10 / 10).
3.  Clean, tokenise, and lemmatise every split using spaCy
    (vocabulary is built from the TRAIN split only → no leakage).
4.  Build a vocabulary capped at 10,000 tokens.
5.  Numericalize + pad / truncate all sequences to length 200.
6.  Wrap each split in a PyTorch DataLoader (batch size 64).
7.  Persist the vocabulary to disk for downstream phases.
8.  Print a concise summary report.

Usage
-----
From the project root::

    python phase1_preprocessing.py

Dependencies
------------
    pip install pandas scikit-learn spacy torch tqdm
    python -m spacy download en_core_web_sm
"""

import logging
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when the script is executed
# directly (not as a module).
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging – human-readable timestamps, level labels, and module names
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "[%(asctime)s] [%(levelname)s] %(name)s – %(message)s",
    datefmt= "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),                  # console
        logging.FileHandler(
            os.path.join(PROJECT_ROOT, "outputs", "phase1.log"),
            mode="w",
            encoding="utf-8",
        ),                                                  # log file
    ],
)

logger = logging.getLogger("phase1")

# ---------------------------------------------------------------------------
# Project imports (after sys.path is fixed)
# ---------------------------------------------------------------------------
from config.config import (
    BATCH_SIZE,
    MAX_SEQ_LEN,
    MAX_VOCAB_SIZE,
    OUTPUTS_DIR,
    RANDOM_STATE,
)
from src.data_loader   import load_data, split_data
from src.preprocessing import preprocess_series
from src.vocabulary    import Vocabulary
from src.numericalize  import numericalize_series
from src.dataset       import create_dataloaders


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_phase1():
    """Execute the full Phase 1 preprocessing pipeline."""

    wall_start = time.time()
    logger.info("=" * 60)
    logger.info("Phase 1 – Data Acquisition & Preprocessing")
    logger.info("=" * 60)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – Load data
    # ------------------------------------------------------------------
    logger.info("STEP 1 ▸ Loading dataset …")
    df = load_data()
    logger.info("  Columns : %s", list(df.columns))
    logger.info("  Shape   : %s", df.shape)

    # ------------------------------------------------------------------
    # Step 2 – Train / Val / Test split  (no further processing yet)
    # ------------------------------------------------------------------
    logger.info("STEP 2 ▸ Splitting dataset  (seed=%d) …", RANDOM_STATE)
    train_df, val_df, test_df = split_data(df)

    logger.info("  Train : %6d samples", len(train_df))
    logger.info("  Val   : %6d samples", len(val_df))
    logger.info("  Test  : %6d samples", len(test_df))

    # Verify class balance in each split
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split["label"].value_counts().to_dict()
        logger.info(
            "  %s class balance → neg: %d | pos: %d",
            name,
            counts.get(0, 0),
            counts.get(1, 0),
        )

    # ------------------------------------------------------------------
    # Step 3 – Clean + Tokenise + Lemmatise
    #          *** FIT-on-TRAIN only for vocabulary (leakage prevention) ***
    # ------------------------------------------------------------------
    logger.info("STEP 3 ▸ Preprocessing text (clean → tokenise → lemmatise) …")

    logger.info("  Processing TRAIN split …")
    t0 = time.time()
    train_tokens = preprocess_series(train_df["review"], show_progress=True)
    logger.info("  Train done in %.1f s", time.time() - t0)

    logger.info("  Processing VAL split …")
    t0 = time.time()
    val_tokens = preprocess_series(val_df["review"], show_progress=True)
    logger.info("  Val done in %.1f s", time.time() - t0)

    logger.info("  Processing TEST split …")
    t0 = time.time()
    test_tokens = preprocess_series(test_df["review"], show_progress=True)
    logger.info("  Test done in %.1f s", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 4 – Build vocabulary  *** TRAIN SPLIT ONLY ***
    # ------------------------------------------------------------------
    logger.info("STEP 4 ▸ Building vocabulary  (max_size=%d, train only) …", MAX_VOCAB_SIZE)
    vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
    vocab.build(train_tokens)  # <-- only training tokens ever touch this call

    logger.info("  Final vocabulary size : %d", len(vocab))
    logger.info(
        "  Top-10 most frequent tokens : %s",
        [w for w, _ in vocab.word_counts.most_common(10)],
    )

    # Persist vocabulary for Phase 2
    vocab_path = vocab.save()
    logger.info("  Vocabulary saved → %s", vocab_path)

    # ------------------------------------------------------------------
    # Step 5 – Numericalize & pad / truncate  (max_len=%d)
    # ------------------------------------------------------------------
    logger.info("STEP 5 ▸ Numericalizing sequences  (max_len=%d) …", MAX_SEQ_LEN)

    train_seqs = numericalize_series(train_tokens, vocab)
    val_seqs   = numericalize_series(val_tokens,   vocab)
    test_seqs  = numericalize_series(test_tokens,  vocab)

    # Quick sanity check
    assert len(train_seqs[0]) == MAX_SEQ_LEN, "Sequence length mismatch!"
    logger.info("  Sequence length check : OK (%d)", MAX_SEQ_LEN)

    # ------------------------------------------------------------------
    # Step 6 – Create PyTorch DataLoaders
    # ------------------------------------------------------------------
    logger.info("STEP 6 ▸ Creating DataLoaders  (batch_size=%d) …", BATCH_SIZE)

    train_labels = train_df["label"].tolist()
    val_labels   = val_df["label"].tolist()
    test_labels  = test_df["label"].tolist()

    train_loader, val_loader, test_loader = create_dataloaders(
        train_seqs,  train_labels,
        val_seqs,    val_labels,
        test_seqs,   test_labels,
    )

    # ------------------------------------------------------------------
    # Step 7 – Verification: inspect one batch from each DataLoader
    # ------------------------------------------------------------------
    logger.info("STEP 7 ▸ Verifying DataLoaders …")

    for name, loader in [
        ("Train", train_loader),
        ("Val",   val_loader),
        ("Test",  test_loader),
    ]:
        batch_seqs, batch_labels = next(iter(loader))
        logger.info(
            "  %s  → sequences: %s | labels: %s",
            name,
            tuple(batch_seqs.shape),
            tuple(batch_labels.shape),
        )
        assert batch_seqs.shape  == (BATCH_SIZE, MAX_SEQ_LEN), \
            f"Unexpected sequence batch shape: {batch_seqs.shape}"
        assert batch_labels.shape == (BATCH_SIZE,), \
            f"Unexpected label batch shape: {batch_labels.shape}"

    # ------------------------------------------------------------------
    # Phase 1 Summary Report
    # ------------------------------------------------------------------
    elapsed = time.time() - wall_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 1 completed in %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("  %-30s %s", "Total samples:", len(df))
    logger.info("  %-30s %s", "Train / Val / Test:",
                f"{len(train_df)} / {len(val_df)} / {len(test_df)}")
    logger.info("  %-30s %d", "Vocabulary size:", len(vocab))
    logger.info("  %-30s %d", "Sequence length:", MAX_SEQ_LEN)
    logger.info("  %-30s %d", "Batch size:", BATCH_SIZE)
    logger.info("  %-30s %d", "Train batches:", len(train_loader))
    logger.info("  %-30s %d", "Val batches:", len(val_loader))
    logger.info("  %-30s %d", "Test batches:", len(test_loader))
    logger.info("  %-30s %s", "Vocabulary saved to:", vocab_path)
    logger.info("  %-30s %s", "Log saved to:", "outputs/phase1.log")
    logger.info("=" * 60)

    # Return artefacts for interactive / notebook use
    return {
        "train_loader" : train_loader,
        "val_loader"   : val_loader,
        "test_loader"  : test_loader,
        "vocab"        : vocab,
        "train_tokens" : train_tokens,
        "val_tokens"   : val_tokens,
        "test_tokens"  : test_tokens,
    }


# ===========================================================================
# Script entry point
# ===========================================================================

if __name__ == "__main__":
    artefacts = run_phase1()
    print("\nPhase 1 complete. Artefacts ready for Phase 2:")
    print(f"  train_loader  → {artefacts['train_loader'].dataset}")
    print(f"  val_loader    → {artefacts['val_loader'].dataset}")
    print(f"  test_loader   → {artefacts['test_loader'].dataset}")
    print(f"  vocab         → {artefacts['vocab']}")
