"""
experiments/utils.py
--------------------
Shared utilities for all experiment scripts.

The most expensive part of Phase 1 is the spaCy tokenisation pass (~5 min
for 50 k reviews).  To avoid re-running it every time an experiment script
is executed, this module provides a caching layer:

  get_phase1_artefacts()
      - If ``outputs/phase1_cache.pkl`` exists: deserialise and return.
      - Otherwise: run the full Phase 1 pipeline, serialise the result,
        then return it.

The cached dict contains:
  train_loader, val_loader, test_loader  – PyTorch DataLoaders
  vocab                                  – fitted Vocabulary object
  train_dataset                          – raw IMDbDataset for K-Fold
"""

import logging
import os
import pickle
import sys

logger = logging.getLogger(__name__)

# Make sure project root is on sys.path when experiments/ scripts are run
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.config import CACHE_PATH


def get_phase1_artefacts() -> dict:
    """
    Return Phase 1 artefacts, using a pickle cache to avoid re-running spaCy.

    Returns
    -------
    dict with keys:
        train_loader  (DataLoader)
        val_loader    (DataLoader)
        test_loader   (DataLoader)
        vocab         (Vocabulary)
        train_dataset (IMDbDataset)

    Notes
    -----
    The cache is stored at ``outputs/phase1_cache.pkl``.
    Delete this file to force a fresh Phase 1 run.
    """
    if os.path.exists(CACHE_PATH):
        logger.info("Loading Phase 1 artefacts from cache: %s", CACHE_PATH)
        with open(CACHE_PATH, "rb") as fh:
            artefacts = pickle.load(fh)
        logger.info(
            "Cache loaded  –  train: %d | val: %d | test: %d | vocab: %d",
            len(artefacts["train_loader"].dataset),
            len(artefacts["val_loader"].dataset),
            len(artefacts["test_loader"].dataset),
            len(artefacts["vocab"]),
        )
        return artefacts

    logger.info("No cache found – running Phase 1 pipeline (this may take ~5 min) …")

    # Run full Phase 1 (imports here to avoid circular deps at module level)
    from phase1_preprocessing import run_phase1

    raw = run_phase1()

    # Attach the raw training Dataset (needed for K-Fold in trainer.py)
    raw["train_dataset"] = raw["train_loader"].dataset

    # Persist to disk
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(raw, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Phase 1 artefacts cached → %s", CACHE_PATH)

    return raw


def setup_logging(arch_name: str) -> None:
    """
    Configure root logger to write to stdout and to
    ``results/<arch_name>/experiment.log``.
    """
    import sys
    from config.config import RESULTS_DIR

    log_dir = os.path.join(RESULTS_DIR, arch_name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level   = logging.INFO,
        format  = "[%(asctime)s] [%(levelname)s] %(name)s – %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir, "experiment.log"),
                mode="w", encoding="utf-8",
            ),
        ],
        force=True,
    )
