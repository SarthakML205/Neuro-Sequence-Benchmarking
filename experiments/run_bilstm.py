"""
experiments/run_bilstm.py
-------------------------
Train and evaluate the Bidirectional LSTM with the best hyperparameters
found during grid search (avg_val_f1 = 0.8708 ± 0.0023).

Usage (from project root)
--------------------------
    python experiments/run_bilstm.py

What this script does
---------------------
1. Loads Phase 1 artefacts (uses pickle cache if available).
2. Trains the BiLSTM on the full training set for N_EPOCHS_FINAL epochs,
   checkpointing the epoch with the highest val F1.
3. Evaluates the best checkpoint on the held-out test set.
4. Saves:
       results/bilstm/best_model.pth
       results/bilstm/experiment.log

Best hyperparameters (from grid search)
----------------------------------------
    hidden_dim : 256
    dropout    : 0.5
    lr         : 1e-4
    optimizer  : adam
    avg_val_f1 : 0.8708 ± 0.0023
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

from config.config       import (
    BATCH_SIZE, EMBED_DIM, GRAD_CLIP, N_EPOCHS_FINAL,
    N_LAYERS, NUM_WORKERS, OUTPUT_DIM, PAD_IDX, RESULTS_DIR,
)
from experiments.utils   import get_phase1_artefacts, setup_logging
from models.bilstm_model import BiLSTMClassifier
from trainer             import evaluate, train_epoch

# ---------------------------------------------------------------------------
# Best hyperparameters (grid search result)
# ---------------------------------------------------------------------------
HIDDEN_DIM: int   = 256
DROPOUT:    float = 0.5
LR:         float = 1e-4
OPTIMIZER:  str   = "adam"

ARCH = "bilstm"


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
        "Hyperparameters → hidden_dim=%d  dropout=%.1f  lr=%g  optimizer=%s",
        HIDDEN_DIM, DROPOUT, LR, OPTIMIZER,
    )

    # ── Build model ─────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
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

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    arch_results_dir = os.path.join(RESULTS_DIR, ARCH)
    os.makedirs(arch_results_dir, exist_ok=True)
    checkpoint_path  = os.path.join(arch_results_dir, "best_model.pth")

    best_val_f1 = -1.0

    logger.info("Training for %d epochs …", N_EPOCHS_FINAL)
    for epoch in range(1, N_EPOCHS_FINAL + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, GRAD_CLIP
        )
        vl_loss, vl_acc, vl_f1 = evaluate(model, val_loader, criterion, device)

        logger.info(
            "  Epoch %2d/%d  tr_loss=%.4f  tr_f1=%.4f  |  "
            "vl_loss=%.4f  vl_acc=%.4f  vl_f1=%.4f",
            epoch, N_EPOCHS_FINAL, tr_loss, tr_f1, vl_loss, vl_acc, vl_f1,
        )

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
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
                    "val_f1":           vl_f1,
                    "val_acc":          vl_acc,
                    "val_loss":         vl_loss,
                },
                checkpoint_path,
            )
            logger.info("  ✔ Best checkpoint saved (val_f1=%.4f)", vl_f1)

    # ── Final evaluation on the held-out test set ────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    ts_loss, ts_acc, ts_f1 = evaluate(model, test_loader, criterion, device)
    logger.info(
        "\n%s FINAL TEST  →  loss=%.4f  acc=%.4f  f1=%.4f",
        ARCH.upper(), ts_loss, ts_acc, ts_f1,
    )

    print(f"\n[{ARCH.upper()}] Experiment complete.")
    print(f"  Test loss : {ts_loss:.4f}")
    print(f"  Test acc  : {ts_acc:.4f}")
    print(f"  Test F1   : {ts_f1:.4f}")
    print(f"  Best model → results/{ARCH}/best_model.pth")
    print(f"  Log        → results/{ARCH}/experiment.log")


if __name__ == "__main__":
    main()
