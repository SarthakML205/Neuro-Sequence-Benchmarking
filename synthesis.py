"""
phase3_synthesis.py
===================
Phase 3 – Synthesis & Visualization for the NLP Comparative Study.

Sections
--------
1. Model Loading      – instantiate each architecture with best hparams,
                        load the saved .pth checkpoint.
2. Benchmark          – evaluate all four models on the held-out test set.
3. Summary Table      – Accuracy, F1, Precision, Recall, Latency.
4. Visualization Suite
     a) Bar Chart      – F1-Score comparison.
     b) Confusion Matrices – 2×2 subplot (one per model).
     c) ROC Curves     – all four on a single graph.
5. Qualitative Error Analysis
                      – 3 reviews where vanilla RNN failed but Bi-LSTM
                        predicted correctly.
6. Deployment Export  – best model + vocabulary → deploy/ folder.

Usage (from project root)
--------------------------
    python phase3_synthesis.py

Outputs
-------
    results/phase3/benchmark_results.csv
    results/phase3/f1_bar_chart.png
    results/phase3/confusion_matrices.png
    results/phase3/roc_curves.png
    deploy/best_model.pth
    deploy/vocabulary.json
    deploy/model_meta.json
"""

import json
import logging
import os
import shutil
import sys
import time

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of execution directory.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from config.config import EMBED_DIM, N_LAYERS, OUTPUT_DIM, OUTPUTS_DIR, RESULTS_DIR
from experiments.utils import get_phase1_artefacts
from models.bilstm_model import BiLSTMClassifier
from models.gru_model import GRUClassifier
from models.lstm_model import LSTMClassifier
from models.rnn_model import RNNClassifier
from src.data_loader import load_data, split_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("phase3")

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
PHASE3_DIR = os.path.join(RESULTS_DIR, "phase3")
DEPLOY_DIR = os.path.join(_ROOT, "deploy")
os.makedirs(PHASE3_DIR, exist_ok=True)
os.makedirs(DEPLOY_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Best hyperparameters per architecture (determined in Phase 2 grid search).
# ---------------------------------------------------------------------------
MODEL_CONFIGS: dict = {
    "RNN": {
        "cls":        RNNClassifier,
        "hidden_dim": 128,
        "dropout":    0.3,
        "checkpoint": os.path.join(RESULTS_DIR, "rnn",    "best_model.pth"),
    },
    "LSTM": {
        "cls":        LSTMClassifier,
        "hidden_dim": 128,
        "dropout":    0.5,
        "checkpoint": os.path.join(RESULTS_DIR, "lstm",   "best_model.pth"),
    },
    "GRU": {
        "cls":        GRUClassifier,
        "hidden_dim": 256,
        "dropout":    0.5,
        "checkpoint": os.path.join(RESULTS_DIR, "gru",    "best_model.pth"),
    },
    "Bi-LSTM": {
        "cls":        BiLSTMClassifier,
        "hidden_dim": 256,
        "dropout":    0.5,
        "checkpoint": os.path.join(RESULTS_DIR, "bilstm", "best_model.pth"),
    },
}

# Consistent color palette across all plots
PALETTE: dict = {
    "RNN":     "#e74c3c",
    "LSTM":    "#3498db",
    "GRU":     "#2ecc71",
    "Bi-LSTM": "#9b59b6",
}


# ===========================================================================
# Section 1 – MODEL LOADING
# ===========================================================================

def build_and_load_model(
    cfg:        dict,
    vocab_size: int,
    device:     torch.device,
) -> nn.Module:
    """
    Instantiate the model from *cfg* and load the saved checkpoint weights.

    Parameters
    ----------
    cfg        : dict         – one entry from MODEL_CONFIGS
    vocab_size : int          – vocabulary size from the fitted Vocabulary
    device     : torch.device

    Returns
    -------
    nn.Module  – model in eval mode, loaded onto *device*
    """
    model = cfg["cls"](
        vocab_size = vocab_size,
        embed_dim  = EMBED_DIM,
        hidden_dim = cfg["hidden_dim"],
        output_dim = OUTPUT_DIM,
        n_layers   = N_LAYERS,
        dropout    = cfg["dropout"],
    )

    # Checkpoints are saved as dicts by trainer.py; extract the state dict.
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        # Fallback: checkpoint is the raw state dict itself
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ===========================================================================
# Section 2 – EVALUATION
# ===========================================================================

def evaluate_model(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Run a full forward pass over *loader* and collect predictions.

    Returns
    -------
    dict with keys:
        logits  : np.ndarray  – raw (pre-sigmoid) model outputs
        probs   : np.ndarray  – sigmoid probabilities in [0, 1]
        preds   : np.ndarray  – binary predictions (threshold = 0.5)
        labels  : np.ndarray  – ground-truth labels
    """
    all_logits: list = []
    all_labels: list = []

    model.eval()
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            logits    = model(sequences).squeeze(1).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))     # numerically-stable sigmoid
    preds  = (probs >= 0.5).astype(int)

    return {"logits": logits, "probs": probs, "preds": preds, "labels": labels}


def measure_latency(
    model:      nn.Module,
    dataset:    torch.utils.data.Dataset,
    device:     torch.device,
    n_samples:  int = 1000,
) -> float:
    """
    Measure inference time for *n_samples* reviews (batched, no grad).

    A single warm-up pass is performed before timing to avoid cold-start
    effects (especially important on GPU).

    Returns
    -------
    float – elapsed time in milliseconds
    """
    n         = min(n_samples, len(dataset))
    sequences = dataset.sequences[:n].to(device)

    model.eval()
    with torch.no_grad():
        # Warm-up
        _ = model(sequences[:1])

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        _  = model(sequences)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1_000

    return elapsed_ms


# ===========================================================================
# Section 3 – SUMMARY TABLE
# ===========================================================================

def build_summary_table(results: dict) -> pd.DataFrame:
    """
    Aggregate per-model metrics into a Pandas DataFrame summary table.

    Columns
    -------
    Accuracy | F1-Score | Precision | Recall | Latency (ms / 1000 samples)
    """
    rows = []
    for name, res in results.items():
        p, l = res["preds"], res["labels"]
        rows.append(
            {
                "Model":                         name,
                "Accuracy":                      round(accuracy_score(l, p),                                           4),
                "F1-Score":                      round(f1_score(l, p, average="binary", zero_division=0),              4),
                "Precision":                     round(precision_score(l, p, average="binary", zero_division=0),       4),
                "Recall":                        round(recall_score(l, p, average="binary", zero_division=0),          4),
                "Latency (ms / 1000 samples)":   round(res["latency_ms"],                                              2),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


# ===========================================================================
# Section 4 – VISUALIZATION SUITE
# ===========================================================================

def plot_f1_bar_chart(summary_df: pd.DataFrame) -> None:
    """Bar chart comparing F1-Scores across all architectures."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [PALETTE.get(m, "#95a5a6") for m in summary_df.index]
    bars   = ax.bar(
        summary_df.index,
        summary_df["F1-Score"],
        color=colors,
        edgecolor="white",
        linewidth=1.5,
        width=0.5,
    )

    # Annotate each bar with its value
    for bar, val in zip(bars, summary_df["F1-Score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    # Dashed line at the best F1 for quick reference
    best_f1 = summary_df["F1-Score"].max()
    ax.axhline(y=best_f1, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title("Test Set F1-Score Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("F1-Score (Binary)")
    ax.set_ylim(0.0, min(summary_df["F1-Score"].max() + 0.1, 1.05))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(PHASE3_DIR, "f1_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Saved F1 bar chart → %s", path)


def plot_confusion_matrices(results: dict) -> None:
    """2×2 subplot of confusion matrices – one per model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes      = axes.ravel()
    fig.suptitle(
        "Confusion Matrices – Test Set",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["labels"], res["preds"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(name, fontsize=13, fontweight="bold",
                     color=PALETTE.get(name, "black"))
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label",      fontsize=10)

    plt.tight_layout()

    path = os.path.join(PHASE3_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Saved confusion matrices → %s", path)


def plot_roc_curves(results: dict) -> None:
    """ROC curves for all four models on a single plot."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Random-classifier baseline
    ax.plot(
        [0, 1], [0, 1],
        "k--", linewidth=1, alpha=0.4,
        label="Random baseline (AUC = 0.50)",
    )

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        roc_auc     = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=PALETTE.get(name, "grey"),
            linewidth=2.5,
            label=f"{name}  (AUC = {roc_auc:.4f})",
        )

    ax.set_title(
        "ROC Curves – All Architectures (Test Set)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(PHASE3_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Saved ROC curves → %s", path)


# ===========================================================================
# Section 5 – QUALITATIVE ERROR ANALYSIS
# ===========================================================================

def qualitative_error_analysis(
    rnn_res:    dict,
    bilstm_res: dict,
    test_texts: list,
    n:          int = 3,
) -> None:
    """
    Print *n* reviews where the vanilla RNN predicted incorrectly but the
    Bi-LSTM predicted correctly, together with confidence scores.

    Confidence for a sample is defined as the model's predicted probability
    for the *true* class (so a higher confidence always means the model is
    more certain about the right answer).

    Parameters
    ----------
    rnn_res    : dict  – output of evaluate_model() for RNN
    bilstm_res : dict  – output of evaluate_model() for Bi-LSTM
    test_texts : list  – raw review strings aligned with test_loader order
    n          : int   – number of examples to display (default 3)
    """
    labels         = rnn_res["labels"]
    rnn_preds      = rnn_res["preds"]
    bilstm_preds   = bilstm_res["preds"]
    rnn_probs      = rnn_res["probs"]
    bilstm_probs   = bilstm_res["probs"]

    # Indices where RNN was wrong AND Bi-LSTM was correct
    error_idxs = [
        i for i in range(len(labels))
        if rnn_preds[i] != labels[i] and bilstm_preds[i] == labels[i]
    ]

    sep  = "─" * 80
    sep2 = "═" * 80

    print(f"\n{sep2}")
    print("  QUALITATIVE ERROR ANALYSIS")
    print(f"  Showing {n} reviews: RNN incorrect  |  Bi-LSTM correct")
    print(f"{sep2}\n")

    if not error_idxs:
        print("  No examples found where RNN failed and Bi-LSTM succeeded.\n")
        return

    for shown, idx in enumerate(error_idxs[:n]):
        true_str  = "Positive" if labels[idx] == 1 else "Negative"

        # Confidence = prob assigned to the *true* class
        rnn_conf    = float(rnn_probs[idx])    if labels[idx] == 1 else 1.0 - float(rnn_probs[idx])
        bilstm_conf = float(bilstm_probs[idx]) if labels[idx] == 1 else 1.0 - float(bilstm_probs[idx])

        rnn_pred_str    = "Positive" if rnn_preds[idx] == 1 else "Negative"
        bilstm_pred_str = "Positive" if bilstm_preds[idx] == 1 else "Negative"

        # Truncate very long reviews to 450 characters for readability
        snippet = test_texts[idx]
        if len(snippet) > 450:
            snippet = snippet[:450] + " …"

        print(sep)
        print(f"  Example {shown + 1} of {n}   |   True Label: {true_str}")
        print(sep)
        print(f"\n  REVIEW:\n  {snippet}\n")
        print(f"  {'Model':<12}  {'Prediction':<12}  {'Confidence (True Class)':>24}  Status")
        print(f"  {'-'*12}  {'-'*12}  {'-'*24}  {'-'*6}")
        print(f"  {'RNN':<12}  {rnn_pred_str:<12}  {rnn_conf:>24.4f}  ✗ Wrong")
        print(f"  {'Bi-LSTM':<12}  {bilstm_pred_str:<12}  {bilstm_conf:>24.4f}  ✓ Correct")
        print()


# ===========================================================================
# Section 6 – DEPLOYMENT EXPORT
# ===========================================================================

def export_for_deployment(summary_df: pd.DataFrame) -> str:
    """
    Export the best-performing model checkpoint and the fitted vocabulary
    to the ``deploy/`` directory for use by a production Streamlit app.

    Files written
    -------------
    deploy/best_model.pth    – model weights (original checkpoint dict)
    deploy/vocabulary.json   – fitted token → index mapping
    deploy/model_meta.json   – architecture name, hparams, and test metrics

    Returns
    -------
    str – name of the winning architecture (e.g. "Bi-LSTM")
    """
    best_arch = summary_df["F1-Score"].idxmax()

    # Derive the folder name: "Bi-LSTM" → "bilstm"
    arch_folder = best_arch.lower().replace("-", "")

    src_ckpt  = os.path.join(RESULTS_DIR, arch_folder, "best_model.pth")
    src_vocab = os.path.join(OUTPUTS_DIR, "vocabulary.json")
    dst_ckpt  = os.path.join(DEPLOY_DIR, "best_model.pth")
    dst_vocab = os.path.join(DEPLOY_DIR, "vocabulary.json")

    shutil.copy2(src_ckpt,  dst_ckpt)
    shutil.copy2(src_vocab, dst_vocab)

    meta = {
        "architecture": best_arch,
        "hidden_dim":   MODEL_CONFIGS[best_arch]["hidden_dim"],
        "dropout":      MODEL_CONFIGS[best_arch]["dropout"],
        "embed_dim":    EMBED_DIM,
        "n_layers":     N_LAYERS,
        "f1_score":     float(summary_df.loc[best_arch, "F1-Score"]),
        "accuracy":     float(summary_df.loc[best_arch, "Accuracy"]),
        "precision":    float(summary_df.loc[best_arch, "Precision"]),
        "recall":       float(summary_df.loc[best_arch, "Recall"]),
    }
    meta_path = os.path.join(DEPLOY_DIR, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("Deployment package written to: %s", DEPLOY_DIR)
    logger.info("  Winner      : %s  (Test F1 = %.4f)", best_arch, meta["f1_score"])
    logger.info("  Files       : best_model.pth · vocabulary.json · model_meta.json")
    return best_arch


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load Phase 1 artefacts (cache-backed to avoid re-running spaCy)
    # ------------------------------------------------------------------
    logger.info("Loading Phase 1 artefacts …")
    artefacts    = get_phase1_artefacts()
    test_loader  = artefacts["test_loader"]
    vocab        = artefacts["vocab"]
    vocab_size   = len(vocab)
    test_dataset = test_loader.dataset
    logger.info("Vocabulary size : %d", vocab_size)
    logger.info("Test set size   : %d samples", len(test_dataset))

    # ------------------------------------------------------------------
    # Recover raw test review texts for qualitative error analysis.
    # The same RANDOM_STATE ensures the split is identical to Phase 1,
    # so test_df rows align in order with test_dataset / test_loader.
    # ------------------------------------------------------------------
    raw_df = load_data()
    _train_df, _val_df, test_df = split_data(raw_df)
    test_texts: list = test_df["review"].tolist()

    assert len(test_texts) == len(test_dataset), (
        f"Mismatch: {len(test_texts)} texts vs {len(test_dataset)} sequences. "
        "Ensure RANDOM_STATE has not changed since Phase 1."
    )

    # ------------------------------------------------------------------
    # 2. Build and load all four models
    # ------------------------------------------------------------------
    logger.info("Loading model checkpoints …")
    models: dict = {}
    for name, cfg in MODEL_CONFIGS.items():
        logger.info("  %-8s ← %s", name, cfg["checkpoint"])
        models[name] = build_and_load_model(cfg, vocab_size, device)

    # ------------------------------------------------------------------
    # 3. Evaluate all models on the held-out test set
    # ------------------------------------------------------------------
    logger.info("Evaluating all models on the test set …")
    results: dict = {}
    for name, model in models.items():
        logger.info("  Evaluating %-8s …", name)
        res               = evaluate_model(model, test_loader, device)
        res["latency_ms"] = measure_latency(model, test_dataset, device)
        results[name]     = res
        logger.info(
            "    Acc=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f  Lat=%.1f ms",
            accuracy_score(res["labels"], res["preds"]),
            f1_score(res["labels"], res["preds"], average="binary", zero_division=0),
            precision_score(res["labels"], res["preds"], average="binary", zero_division=0),
            recall_score(res["labels"], res["preds"], average="binary", zero_division=0),
            res["latency_ms"],
        )

    # ------------------------------------------------------------------
    # 4. Build and print the summary table
    # ------------------------------------------------------------------
    summary_df = build_summary_table(results)

    print("\n" + "═" * 68)
    print("  PHASE 3 – FINAL BENCHMARK RESULTS (Test Set)")
    print("═" * 68)
    print(summary_df.to_string())
    print()

    # ------------------------------------------------------------------
    # 5. Visualization suite
    # ------------------------------------------------------------------
    logger.info("Generating visualizations …")
    plot_f1_bar_chart(summary_df)
    plot_confusion_matrices(results)
    plot_roc_curves(results)

    # ------------------------------------------------------------------
    # 6. Qualitative error analysis
    # ------------------------------------------------------------------
    qualitative_error_analysis(
        rnn_res    = results["RNN"],
        bilstm_res = results["Bi-LSTM"],
        test_texts = test_texts,
    )

    # ------------------------------------------------------------------
    # 7. Export best model for deployment
    # ------------------------------------------------------------------
    logger.info("Exporting best model for deployment …")
    winner = export_for_deployment(summary_df)

    # ------------------------------------------------------------------
    # 8. Persist summary table to CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(PHASE3_DIR, "benchmark_results.csv")
    summary_df.to_csv(csv_path)
    logger.info("Benchmark table saved → %s", csv_path)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("═" * 68)
    print(f"  Phase 3 complete.  Overall winner: {winner}")
    print(f"  Plots        → {PHASE3_DIR}")
    print(f"  Deploy pkg   → {DEPLOY_DIR}")
    print("═" * 68)


if __name__ == "__main__":
    main()
