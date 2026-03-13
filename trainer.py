"""
trainer.py
----------
Shared experiment harness for Phase 2. The training logic is architecture-
agnostic: all four models (RNN / LSTM / GRU / BiLSTM) are driven by the
same functions, ensuring the only experimental variable is the recurrent
layer.

Public API
----------
train_epoch         – one full pass over a DataLoader (with gradient clipping)
evaluate            – forward-only pass, returns loss + metrics
run_kfold           – K-Fold cross-validation over a train Dataset
hyperparameter_search
                    – grid search → K-Fold → retrain best → checkpoint

Metrics
-------
  • Binary Cross-Entropy loss  (nn.BCEWithLogitsLoss)
  • Accuracy
  • F1-Score  (binary, positive class)  ← primary ranking metric

Notes on design
---------------
* BCEWithLogitsLoss fuses sigmoid + BCE for numerical stability.
  The model must therefore return raw *logits* (no sigmoid in forward()).
* Gradient clipping is applied inside train_epoch with max_norm=GRAD_CLIP.
* The best model across all folds/configs is saved to
    results/<arch_name>/best_model.pth
  along with a summary CSV.
"""

import itertools
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config.config import (
    BATCH_SIZE,
    GRAD_CLIP,
    N_EPOCHS_FINAL,
    N_EPOCHS_SEARCH,
    N_FOLDS,
    NUM_WORKERS,
    PARAM_GRID,
    RESULTS_DIR,
    EMBED_DIM,
    N_LAYERS,
    OUTPUT_DIM,
    PAD_IDX,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================

def _build_optimizer(
    name: str,
    params,
    lr: float,
) -> torch.optim.Optimizer:
    """
    Create an Adam or RMSprop optimiser.

    Parameters
    ----------
    name   : str         – "adam" or "rmsprop" (case-insensitive)
    params              – model.parameters()
    lr     : float       – learning rate

    Returns
    -------
    torch.optim.Optimizer
    """
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "rmsprop":
        # alpha=0.99 and momentum=0.0 match the Keras/TF defaults that are
        # commonly reported to work well for RNNs.
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, momentum=0.0)
    else:
        raise ValueError(f"Unsupported optimizer: '{name}'. Use 'adam' or 'rmsprop'.")


def _compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute accuracy and binary F1-score from raw logits.

    Parameters
    ----------
    logits : Tensor – (N,) or (N, 1), raw model output (no sigmoid)
    labels : Tensor – (N,) binary integer labels

    Returns
    -------
    (accuracy, f1) : Tuple[float, float]
    """
    probs = torch.sigmoid(logits.squeeze()).cpu()
    preds = (probs >= 0.5).long().numpy()
    truth = labels.cpu().numpy()

    acc = accuracy_score(truth, preds)
    f1  = f1_score(truth, preds, average="binary", zero_division=0)
    return float(acc), float(f1)


# ===========================================================================
# Resume helpers
# ===========================================================================

_PROGRESS_FILE  = "grid_progress.json"     # saved inside arch results dir
_RESUME_FILE    = "final_retrain_resume.pth" # saved inside arch results dir


def _combo_key(combo: dict) -> str:
    """
    Stable string key for a hyperparameter combo dict.
    Two combos with the same values always produce the same key regardless
    of dict insertion order.
    """
    return json.dumps(dict(sorted(combo.items())), sort_keys=True)


def _load_grid_progress(results_dir: str) -> List[dict]:
    """
    Load previously completed combo records from ``grid_progress.json``.
    Returns an empty list if the file does not exist or is corrupt.
    """
    path = os.path.join(results_dir, _PROGRESS_FILE)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info(
            "Resume: loaded %d completed combo(s) from %s", len(data), path
        )
        return data
    except (json.JSONDecodeError, KeyError):
        logger.warning("Could not parse %s – starting grid search fresh.", path)
        return []


def _save_grid_progress(records: List[dict], results_dir: str) -> None:
    """Atomically persist the current records list to ``grid_progress.json``."""
    path = os.path.join(results_dir, _PROGRESS_FILE)
    # Write to a temp file then rename for atomicity
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    os.replace(tmp, path)


# ===========================================================================
# Core training / evaluation
# ===========================================================================

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    clip:      float = GRAD_CLIP,
) -> Tuple[float, float, float]:
    """
    Run one full training epoch.

    Strategy
    --------
    * Mixed-precision (AMP) is intentionally omitted to keep the code
      readable; add torch.cuda.amp.autocast if you need GPU speed.
    * Gradient clipping (``clip_grad_norm_``) prevents exploding gradients
      which are particularly common in vanilla RNNs.

    Parameters
    ----------
    model     : nn.Module           – PyTorch model in training mode
    loader    : DataLoader          – training data
    optimizer : Optimizer
    criterion : nn.Module           – BCEWithLogitsLoss
    device    : torch.device
    clip      : float               – max gradient norm

    Returns
    -------
    (avg_loss, accuracy, f1) : Tuple[float, float, float]
    """
    model.train()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels    = labels.to(device).float()

        optimizer.zero_grad()

        logits = model(sequences).squeeze(1)   # (batch,)
        loss   = criterion(logits, labels)

        loss.backward()

        # Gradient clipping – critical for RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach().long())

    avg_loss   = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc, f1    = _compute_metrics(all_logits, all_labels)

    return avg_loss, acc, f1


def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate a model on a DataLoader without updating gradients.

    Parameters
    ----------
    model     : nn.Module  – in eval mode after this call
    loader    : DataLoader
    criterion : nn.Module
    device    : torch.device

    Returns
    -------
    (avg_loss, accuracy, f1) : Tuple[float, float, float]
    """
    model.eval()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels    = labels.to(device).float()

            logits = model(sequences).squeeze(1)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * sequences.size(0)
            all_logits.append(logits)
            all_labels.append(labels.long())

    avg_loss   = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc, f1    = _compute_metrics(all_logits, all_labels)

    return avg_loss, acc, f1


# ===========================================================================
# K-Fold Cross-Validation
# ===========================================================================

def run_kfold(
    model_class,
    model_kwargs:   Dict[str, Any],
    train_dataset,
    optimizer_type: str,
    lr:             float,
    device:         torch.device,
    n_folds:        int   = N_FOLDS,
    n_epochs:       int   = N_EPOCHS_SEARCH,
    batch_size:     int   = BATCH_SIZE,
    clip:           float = GRAD_CLIP,
) -> Dict[str, float]:
    """
    K-Fold CV over ``train_dataset`` for a single (model, hyperparams) combo.

    A fresh model + optimiser is created for *each fold* to ensure fold
    results are independent.

    Parameters
    ----------
    model_class   : type               – model class (e.g. LSTMClassifier)
    model_kwargs  : dict               – kwargs for model __init__
    train_dataset : IMDbDataset        – *full* training dataset
    optimizer_type: str                – "adam" or "rmsprop"
    lr            : float              – learning rate
    device        : torch.device
    n_folds       : int                – K in K-Fold
    n_epochs      : int                – epochs per fold
    batch_size    : int
    clip          : float              – gradient clip max_norm

    Returns
    -------
    dict with keys: avg_val_loss, avg_val_acc, avg_val_f1,
                    std_val_f1  (std-dev across folds)
    """
    criterion = nn.BCEWithLogitsLoss()
    kf        = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    indices   = list(range(len(train_dataset)))

    fold_f1s:  List[float] = []
    fold_accs: List[float] = []
    fold_losses: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        # Subset datasets for this fold
        fold_train_ds = Subset(train_dataset, train_idx)
        fold_val_ds   = Subset(train_dataset, val_idx)

        fold_train_loader = DataLoader(
            fold_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS,
        )
        fold_val_loader = DataLoader(
            fold_val_ds, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS,
        )

        # Fresh model and optimizer for each fold
        model     = model_class(**model_kwargs).to(device)
        optimizer = _build_optimizer(optimizer_type, model.parameters(), lr)

        best_val_f1 = 0.0

        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc, train_f1 = train_epoch(
                model, fold_train_loader, optimizer, criterion, device, clip
            )
            val_loss, val_acc, val_f1 = evaluate(
                model, fold_val_loader, criterion, device
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            logger.debug(
                "  Fold %d/%d  Epoch %d/%d  "
                "train_loss=%.4f  val_loss=%.4f  val_f1=%.4f",
                fold_idx, n_folds, epoch, n_epochs,
                train_loss, val_loss, val_f1,
            )

        fold_f1s.append(best_val_f1)
        fold_accs.append(val_acc)
        fold_losses.append(val_loss)
        logger.info(
            "    Fold %d/%d  best_val_f1=%.4f", fold_idx, n_folds, best_val_f1
        )

    avg_f1  = sum(fold_f1s)  / n_folds
    std_f1  = float(pd.Series(fold_f1s).std())
    avg_acc = sum(fold_accs) / n_folds
    avg_loss= sum(fold_losses) / n_folds

    return {
        "avg_val_loss": avg_loss,
        "avg_val_acc":  avg_acc,
        "avg_val_f1":   avg_f1,
        "std_val_f1":   std_f1,
    }


# ===========================================================================
# Hyperparameter grid search + final retraining
# ===========================================================================

def hyperparameter_search(
    model_class,
    arch_name:       str,
    train_dataset,
    val_loader:      DataLoader,
    test_loader:     DataLoader,
    vocab_size:      int,
    param_grid:      Optional[Dict[str, List[Any]]] = None,
    device:          Optional[torch.device]          = None,
    n_folds:         int   = N_FOLDS,
    n_epochs_search: int   = N_EPOCHS_SEARCH,
    n_epochs_final:  int   = N_EPOCHS_FINAL,
    batch_size:      int   = BATCH_SIZE,
    clip:            float = GRAD_CLIP,
) -> Tuple[nn.Module, pd.DataFrame]:
    """
    Full hyperparameter search for one architecture.

    Algorithm
    ---------
    1. Enumerate all combinations in ``param_grid``.
    2. For each combo run K-Fold CV on ``train_dataset``.
    3. Select the combo with the highest ``avg_val_f1``.
    4. Retrain the best combo on the *entire* ``train_dataset`` for
       ``n_epochs_final`` epochs, monitoring val F1 for checkpointing.
    5. Evaluate on ``test_loader`` and report final metrics.
    6. Save:
         results/<arch_name>/best_model.pth
         results/<arch_name>/tuning_results.csv

    Parameters
    ----------
    model_class      : type          – model class
    arch_name        : str           – e.g. "lstm"  (used for folder naming)
    train_dataset    : IMDbDataset
    val_loader       : DataLoader
    test_loader      : DataLoader
    vocab_size       : int
    param_grid       : dict | None   – defaults to PARAM_GRID from config
    device           : torch.device | None – defaults to CUDA if available
    n_folds          : int
    n_epochs_search  : int
    n_epochs_final   : int
    batch_size       : int
    clip             : float

    Returns
    -------
    (best_model, results_df) : Tuple[nn.Module, pd.DataFrame]
        best_model  – retrained model with best hyperparameters
        results_df  – full grid-search results table
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if param_grid is None:
        param_grid = PARAM_GRID

    arch_results_dir = os.path.join(RESULTS_DIR, arch_name)
    os.makedirs(arch_results_dir, exist_ok=True)

    # Fixed model kwargs (do not vary per combo)
    base_kwargs = dict(
        vocab_size = vocab_size,
        embed_dim  = EMBED_DIM,
        output_dim = OUTPUT_DIM,
        n_layers   = N_LAYERS,
        pad_idx    = PAD_IDX,
    )

    # Build the Cartesian product of the grid
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total  = len(combos)

    # ------------------------------------------------------------------
    # Resume: load any previously completed combo records
    # ------------------------------------------------------------------
    records      = _load_grid_progress(arch_results_dir)
    done_keys    = {_combo_key({k: r[k] for k in keys}) for r in records}
    remaining    = total - len(done_keys)

    # Restore best_val_f1 / best_params from already-completed records
    best_val_f1 = -1.0
    best_params = {}
    for r in records:
        if r["avg_val_f1"] > best_val_f1:
            best_val_f1 = r["avg_val_f1"]
            best_params = {k: r[k] for k in keys}

    logger.info(
        "=" * 60 + "\n"
        "  Architecture : %s\n"
        "  Device       : %s\n"
        "  Grid combos  : %d total  |  %d done  |  %d remaining\n"
        "  (×%d-Fold CV, %d epochs/fold)\n"
        + "=" * 60,
        arch_name.upper(), device, total, len(done_keys), remaining,
        n_folds, n_epochs_search,
    )

    for combo_idx, combo_vals in enumerate(combos, start=1):
        combo = dict(zip(keys, combo_vals))

        # ── SKIP if this combo was already completed in a previous run ──
        if _combo_key(combo) in done_keys:
            logger.info(
                "[%d/%d]  SKIP (already done)  %s",
                combo_idx, total,
                "  ".join(f"{k}={v}" for k, v in combo.items()),
            )
            continue

        logger.info(
            "[%d/%d]  %s", combo_idx, total,
            "  ".join(f"{k}={v}" for k, v in combo.items()),
        )
        t0 = time.time()

        # Build model kwargs for this combo
        model_kwargs = {
            **base_kwargs,
            "hidden_dim": combo["hidden_dim"],
            "dropout":    combo["dropout"],
        }

        cv_results = run_kfold(
            model_class    = model_class,
            model_kwargs   = model_kwargs,
            train_dataset  = train_dataset,
            optimizer_type = combo["optimizer"],
            lr             = combo["lr"],
            device         = device,
            n_folds        = n_folds,
            n_epochs       = n_epochs_search,
            batch_size     = batch_size,
            clip           = clip,
        )

        elapsed = time.time() - t0
        record  = {**combo, **cv_results, "elapsed_s": round(elapsed, 1)}
        records.append(record)

        # ── PERSIST after every completed combo ──────────────────────────
        _save_grid_progress(records, arch_results_dir)

        logger.info(
            "         avg_val_f1=%.4f ± %.4f  (%.1fs)",
            cv_results["avg_val_f1"], cv_results["std_val_f1"], elapsed,
        )

        if cv_results["avg_val_f1"] > best_val_f1:
            best_val_f1 = cv_results["avg_val_f1"]
            best_params = combo.copy()

    # ------------------------------------------------------------------
    # Retrain the best configuration on the FULL training set
    # ------------------------------------------------------------------
    logger.info(
        "\nBest config  →  %s  (avg_val_f1=%.4f)",
        best_params, best_val_f1,
    )
    logger.info(
        "Retraining on full train set for %d epochs …", n_epochs_final
    )

    best_model_kwargs = {
        **base_kwargs,
        "hidden_dim": best_params["hidden_dim"],
        "dropout":    best_params["dropout"],
    }
    best_model    = model_class(**best_model_kwargs).to(device)
    optimizer     = _build_optimizer(
        best_params["optimizer"], best_model.parameters(), best_params["lr"]
    )
    criterion     = nn.BCEWithLogitsLoss()

    full_train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS,
    )

    checkpoint_path = os.path.join(arch_results_dir, "best_model.pth")
    resume_path     = os.path.join(arch_results_dir, _RESUME_FILE)

    # ── Resume final training if a mid-run state exists ─────────────────
    start_epoch        = 1
    best_checkpoint_f1 = -1.0

    if os.path.exists(resume_path):
        try:
            resume_ckpt = torch.load(resume_path, map_location=device)
            best_model.load_state_dict(resume_ckpt["model_state_dict"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            start_epoch        = resume_ckpt["epoch"] + 1
            best_checkpoint_f1 = resume_ckpt.get("best_checkpoint_f1", -1.0)
            logger.info(
                "Resume: continuing final training from epoch %d/%d  "
                "(best_checkpoint_f1 so far=%.4f)",
                start_epoch, n_epochs_final, best_checkpoint_f1,
            )
        except Exception as exc:  # corrupt / incompatible → restart
            logger.warning("Could not load %s (%s) – restarting final training.", resume_path, exc)
            start_epoch = 1

    if start_epoch > n_epochs_final:
        logger.info("Final training already complete – loading best checkpoint.")
    else:
        pass  # will enter the loop below

    for epoch in tqdm(
        range(start_epoch, n_epochs_final + 1),
        desc=f"  Final training [{arch_name}]",
        initial=start_epoch - 1,
        total=n_epochs_final,
    ):
        train_loss, train_acc, train_f1 = train_epoch(
            best_model, full_train_loader, optimizer, criterion, device, clip
        )
        val_loss, val_acc, val_f1 = evaluate(
            best_model, val_loader, criterion, device
        )

        logger.info(
            "  Epoch %2d/%d  "
            "train_loss=%.4f  train_f1=%.4f  |  "
            "val_loss=%.4f  val_acc=%.4f  val_f1=%.4f",
            epoch, n_epochs_final,
            train_loss, train_f1, val_loss, val_acc, val_f1,
        )

        # ── Always save a resume state so we can continue if interrupted ──
        torch.save(
            {
                "epoch":              epoch,
                "model_state_dict":   best_model.state_dict(),
                "optimizer_state":    optimizer.state_dict(),
                "best_checkpoint_f1": best_checkpoint_f1,
            },
            resume_path,
        )

        # ── Checkpoint the BEST val-F1 model separately ──────────────────
        if val_f1 > best_checkpoint_f1:
            best_checkpoint_f1 = val_f1
            torch.save(
                {
                    "epoch":            epoch,
                    "arch":             arch_name,
                    "best_params":      best_params,
                    "model_state_dict": best_model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "val_f1":           val_f1,
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                },
                checkpoint_path,
            )
            logger.info("  ✔ Best checkpoint saved (val_f1=%.4f)", val_f1)

    # Final training complete – remove the resume state file
    if os.path.exists(resume_path):
        os.remove(resume_path)
        logger.info("Resume file removed (final training complete).")

    # ------------------------------------------------------------------
    # Final evaluation on the held-out test set
    # ------------------------------------------------------------------
    # Load the best checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    best_model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_f1 = evaluate(
        best_model, test_loader, criterion, device
    )

    logger.info(
        "\n%s FINAL TEST  →  loss=%.4f  acc=%.4f  f1=%.4f",
        arch_name.upper(), test_loss, test_acc, test_f1,
    )

    # Update the checkpoint with test metrics
    ckpt["test_loss"] = test_loss
    ckpt["test_acc"]  = test_acc
    ckpt["test_f1"]   = test_f1
    torch.save(ckpt, checkpoint_path)

    # ------------------------------------------------------------------
    # Persist tuning results table
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values("avg_val_f1", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(arch_results_dir, "tuning_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Tuning results saved → %s", csv_path)
    logger.info("Best model saved     → %s", checkpoint_path)

    return best_model, results_df
