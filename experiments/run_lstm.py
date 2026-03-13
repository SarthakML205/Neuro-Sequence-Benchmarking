"""
experiments/run_lstm.py
-----------------------
Trains the LSTM classifier with the best hyperparameters found during
grid search (avg_val_f1 = 0.8648 ± 0.0047).

Best configuration
------------------
    hidden_dim : 128
    dropout    : 0.5
    lr         : 0.0001
    optimizer  : adam

Usage (from project root)
--------------------------
    python experiments/run_lstm.py

What this script does
---------------------
1. Loads Phase 1 artefacts (uses pickle cache if available).
2. Builds the LSTM model with the best hyperparameters.
3. Trains for N_EPOCHS_FINAL epochs on the full training set,
   checkpointing the best val-F1 model.
4. Loads the best checkpoint and evaluates on the held-out test set.
5. Saves:
     results/lstm/best_model.pth
     results/lstm/experiment.log

LSTM-specific notes
-------------------
* The forget-gate bias is initialised to 1 in LSTMClassifier._init_weights(),
  which helps the model remember long-range context from the very first epoch.
* LSTMs are generally more stable than vanilla RNNs but may still benefit
  from gradient clipping when stacked (n_layers=2).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import (
    BATCH_SIZE,
    EMBED_DIM,
    GRAD_CLIP,
    N_EPOCHS_FINAL,
    N_LAYERS,
    NUM_WORKERS,
    OUTPUT_DIM,
    PAD_IDX,
    RESULTS_DIR,
)
from experiments.utils  import get_phase1_artefacts, setup_logging
from models.lstm_model  import LSTMClassifier
from trainer            import evaluate, train_epoch

# ---------------------------------------------------------------------------
# Best hyperparameters (grid search: avg_val_f1 = 0.8648 ± 0.0047)
# ---------------------------------------------------------------------------
HIDDEN_DIM: int   = 128
DROPOUT:    float = 0.5
LR:         float = 1e-4
OPTIMIZER:  str   = "adam"

ARCH = "lstm"


def main() -> None:
    setup_logging(ARCH)
    logger = logging.getLogger(ARCH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    artefacts     = get_phase1_artefacts()
    train_dataset = artefacts["train_dataset"]
    val_loader    = artefacts["val_loader"]
    test_loader   = artefacts["test_loader"]
    vocab_size    = len(artefacts["vocab"])

    logger.info("Vocab size : %d", vocab_size)
    logger.info(
        "Config  →  hidden_dim=%d  dropout=%.1f  lr=%.0e  optimizer=%s",
        HIDDEN_DIM, DROPOUT, LR, OPTIMIZER,
    )

    model = LSTMClassifier(
        vocab_size = vocab_size,
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        output_dim = OUTPUT_DIM,
        n_layers   = N_LAYERS,
        dropout    = DROPOUT,
        pad_idx    = PAD_IDX,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    full_train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
    )

    arch_results_dir = os.path.join(RESULTS_DIR, ARCH)
    os.makedirs(arch_results_dir, exist_ok=True)
    checkpoint_path = os.path.join(arch_results_dir, "best_model.pth")

    best_val_f1 = -1.0

    logger.info("Training for %d epochs …", N_EPOCHS_FINAL)
    for epoch in tqdm(range(1, N_EPOCHS_FINAL + 1), desc=f"  Training [{ARCH}]"):
        train_loss, train_acc, train_f1 = train_epoch(
            model, full_train_loader, optimizer, criterion, device, GRAD_CLIP
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        logger.info(
            "  Epoch %2d/%d  "
            "train_loss=%.4f  train_f1=%.4f  |  "
            "val_loss=%.4f  val_acc=%.4f  val_f1=%.4f",
            epoch, N_EPOCHS_FINAL,
            train_loss, train_f1, val_loss, val_acc, val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch":            epoch,
                    "arch":             ARCH,
                    "hidden_dim":       HIDDEN_DIM,
                    "dropout":          DROPOUT,
                    "lr":               LR,
                    "optimizer":        OPTIMIZER,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "val_f1":           val_f1,
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                },
                checkpoint_path,
            )
            logger.info("  ✔ Best checkpoint saved (val_f1=%.4f)", val_f1)

    # Evaluate best checkpoint on test set
    logger.info("\nLoading best checkpoint for test evaluation …")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    logger.info(
        "Test results  →  loss=%.4f  acc=%.4f  f1=%.4f",
        test_loss, test_acc, test_f1,
    )

    print(f"\n[{ARCH.upper()}] Experiment complete.")
    print(f"  Best model → results/{ARCH}/best_model.pth")
    print(f"  Log        → results/{ARCH}/experiment.log")
    print(f"  Test F1    : {test_f1:.4f}")
    print(f"  Test Acc   : {test_acc:.4f}")


if __name__ == "__main__":
    main()
