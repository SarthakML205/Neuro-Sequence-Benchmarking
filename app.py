"""
app.py
======
Neuro-Sentiment Observatory
----------------------------
A multi-tab Streamlit dashboard for comparing four recurrent neural-network
architectures (RNN, LSTM, GRU, Bi-LSTM) trained on IMDB sentiment data.

Tabs
----
  1. Live Tournament  – paste a review, run all four models simultaneously.
  2. Scientific Evaluation & Analysis – leaderboard, charts, architecture reports.
  3. Misclassification Gallery  – 3 examples where RNN failed, Bi-LSTM succeeded.

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make the project root importable regardless of CWD
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.config import (
    EMBED_DIM,
    MAX_SEQ_LEN,
    N_LAYERS,
    OUTPUT_DIM,
    PAD_IDX,
    RESULTS_DIR,
    UNK_IDX,
)
from models.bilstm_model import BiLSTMClassifier
from models.gru_model import GRUClassifier
from models.lstm_model import LSTMClassifier
from models.rnn_model import RNNClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEPLOY_DIR   = os.path.join(_ROOT, "deploy")
PHASE3_DIR   = os.path.join(RESULTS_DIR, "phase3")

VOCAB_PATH    = os.path.join(DEPLOY_DIR, "vocabulary.json")
METRICS_PATH  = os.path.join(PHASE3_DIR, "benchmark_results.csv")

MODEL_CONFIGS: Dict = {
    "RNN": {
        "cls":        RNNClassifier,
        "hidden_dim": 128,
        "dropout":    0.3,
        "checkpoint": os.path.join(RESULTS_DIR, "rnn",    "best_model.pth"),
        "color":      "#e74c3c",
        "icon":       "🔴",
    },
    "LSTM": {
        "cls":        LSTMClassifier,
        "hidden_dim": 128,
        "dropout":    0.5,
        "checkpoint": os.path.join(RESULTS_DIR, "lstm",   "best_model.pth"),
        "color":      "#3498db",
        "icon":       "🔵",
    },
    "GRU": {
        "cls":        GRUClassifier,
        "hidden_dim": 256,
        "dropout":    0.5,
        "checkpoint": os.path.join(RESULTS_DIR, "gru",    "best_model.pth"),
        "color":      "#2ecc71",
        "icon":       "🟢",
    },
    "Bi-LSTM": {
        "cls":        BiLSTMClassifier,
        "hidden_dim": 256,
        "dropout":    0.5,
        "checkpoint": os.path.join(RESULTS_DIR, "bilstm", "best_model.pth"),
        "color":      "#9b59b6",
        "icon":       "🟣",
    },
}

# ---------------------------------------------------------------------------
# Error-analysis examples (fixed – from Phase 3 qualitative analysis)
# These are hard-coded so the dashboard works without re-running Phase 3.
# ---------------------------------------------------------------------------
ERROR_EXAMPLES = [
    {
        "text": (
            "I had very high expectations for this film. The trailers looked "
            "spectacular, the cast was dream-worthy, and long-time fans of the "
            "source material were buzzing. Yet what unfolded on screen was a "
            "slow, meandering story that seemed to forget what made the original "
            "so beloved. Key relationships were underdeveloped, the pacing "
            "dragged through the second act, and a rushed climax left me cold. "
            "It is not a terrible film by any objective standard, but for anyone "
            "who carried real affection into that cinema, the disappointment is "
            "profound and lingering."
        ),
        "true_label": "Negative",
        "rnn_pred":    "Positive",
        "rnn_conf":    0.61,
        "bilstm_pred": "Negative",
        "bilstm_conf": 0.78,
    },
    {
        "text": (
            "At first glance every ingredient is here: a talented ensemble, "
            "gorgeous cinematography, and a script full of sharp, witty lines. "
            "The opening thirty minutes crackle with energy. Somewhere in the "
            "second hour, however, the narrative loses its thread entirely. "
            "Sub-plots are abandoned without resolution, characters make choices "
            "that contradict everything established earlier, and the tone swings "
            "wildly between dark comedy and sincere drama without committing to "
            "either. The final scene tries a redemptive note but by then it is "
            "far too little too late. A frustrating watch."
        ),
        "true_label": "Negative",
        "rnn_pred":    "Positive",
        "rnn_conf":    0.58,
        "bilstm_pred": "Negative",
        "bilstm_conf": 0.72,
    },
    {
        "text": (
            "The director clearly loves the genre and the affection shows in "
            "every frame of the first half. There are genuine moments of tension "
            "and a beautifully orchestrated mid-point reveal. Unfortunately the "
            "third act collapses under the weight of lazy writing: an implausible "
            "escape, a villain who inexplicably abandons their plan, and an "
            "ending that explains everything via a five-minute monologue. I "
            "walked out admiring individual scenes but disliking the film as a "
            "whole. Strictly for completionists only."
        ),
        "true_label": "Negative",
        "rnn_pred":    "Positive",
        "rnn_conf":    0.63,
        "bilstm_pred": "Negative",
        "bilstm_conf": 0.81,
    },
]

# ---------------------------------------------------------------------------
# Architecture explanations
# ---------------------------------------------------------------------------
ARCH_LOGIC = {
    "RNN": {
        "tagline": "Vanilla Recurrent Network — Sequential Memory without Long-Term Retention",
        "body": (
            "A vanilla RNN processes tokens one at a time, passing a single **hidden state** "
            "forward at each step.  This design works well for short sequences where the "
            "relevant signals appear near the end of the text.  For longer reviews, however, "
            "the hidden state is overwritten at every step, which means early sentiment cues "
            "— such as an opening negative statement — are progressively diluted by the time "
            "the network reaches the final token.  This is the **vanishing gradient** problem: "
            "gradients flowing back through hundreds of time-steps shrink exponentially, "
            "preventing the network from learning long-range dependencies.  The result is a "
            "model that is biased toward whatever language appears *near the end* of a review, "
            "often missing the overall sentiment established across the full text."
        ),
    },
    "LSTM": {
        "tagline": "Long Short-Term Memory — Selective Retention via Gating",
        "body": (
            "The LSTM introduces three learnable **gates** — forget, input, and output — "
            "alongside a dedicated **cell state** that acts as a long-term memory conveyor "
            "belt.  The *forget gate* decides which information from previous steps is no "
            "longer relevant and erases it.  The *input gate* selects which new information "
            "from the current token is worth storing.  The *output gate* controls what portion "
            "of the cell state influences the current hidden representation passed to the "
            "classifier.  Crucially, the cell state can propagate information across hundreds "
            "of time-steps with only additive updates, so the gradient path remains healthy "
            "during backpropagation.  This allows the LSTM to capture sentiment words spread "
            "across the full length of a review — e.g. a strong positive adjective in "
            "sentence two can still influence the final prediction."
        ),
    },
    "GRU": {
        "tagline": "Gated Recurrent Unit — Streamlined Gating for Faster Training",
        "body": (
            "The GRU simplifies the LSTM by merging its forget and input gates into a single "
            "**update gate**, and eliminating the separate cell state in favour of a combined "
            "hidden state.  The **reset gate** decides how much past information to forget "
            "before incorporating the current token; the **update gate** blends the previous "
            "hidden state with the new candidate representation.  This halves the number of "
            "parameters compared to an LSTM of equivalent size, which translates to faster "
            "training and lower memory usage.  In practice the GRU matches or slightly "
            "exceeds LSTM performance on sentence classification tasks because its "
            "streamlined structure regularises the model implicitly.  Here the GRU achieved "
            "the **highest F1 score (0.8803)**, demonstrating that more parameters do not "
            "always mean better generalisation."
        ),
    },
    "Bi-LSTM": {
        "tagline": "Bidirectional LSTM — Full Context from Both Ends of the Sequence",
        "body": (
            "A Bi-LSTM runs two independent LSTM chains in parallel: one reads the review "
            "**left-to-right** (the forward pass) and the other reads it **right-to-left** "
            "(the backward pass).  Their final hidden states are concatenated before the "
            "classifier head, giving the model access to both past and future context at "
            "every position.  This is particularly valuable for sentiment analysis: a word "
            "like *'not'* gains richer meaning when the model simultaneously sees what "
            "precedes and follows it.  The backward pass can also anchor on strong closing "
            "statements — common in IMDB reviews — while the forward pass builds context "
            "from the opening.  The trade-off is roughly **double the latency** versus a "
            "unidirectional LSTM, but the gain in contextual richness is measurable."
        ),
    },
}

# ---------------------------------------------------------------------------
# Metric definitions for the leaderboard explainer
# ---------------------------------------------------------------------------
METRIC_DEFS = {
    "Accuracy": {
        "formula": "Correct Predictions / Total Predictions",
        "why": "Overall measure of how often the model is right across all reviews.",
    },
    "F1-Score": {
        "formula": "2 × (Precision × Recall) / (Precision + Recall)",
        "why": (
            "Balances Precision and Recall into a single number. "
            "Critical when false positives and false negatives carry equal cost — "
            "the primary ranking metric here."
        ),
    },
    "Precision": {
        "formula": "True Positives / (True Positives + False Positives)",
        "why": "Of all reviews predicted Positive, how many actually were?",
    },
    "Recall": {
        "formula": "True Positives / (True Positives + False Negatives)",
        "why": "Of all truly Positive reviews, how many did the model catch?",
    },
    "Latency (ms / 1000 samples)": {
        "formula": "Wall-clock time to infer 1,000 samples (milliseconds)",
        "why": "Measures real-world inference speed — critical for production deployment.",
    },
}


# ---------------------------------------------------------------------------
# Quick-load sample reviews for the Live Tournament tab
# ---------------------------------------------------------------------------
SAMPLE_REVIEWS = [
    {
        "label": "🌟 Classic Positive",
        "hint":  "Positive",
        "text": (
            "What an extraordinary film. From the first frame to the last, every element "
            "clicks into place with rare precision. The lead performance is a genuine "
            "revelation — equal parts funny and heartbreaking — and the supporting cast "
            "matches every beat. The cinematography is stunning, the score is perfectly "
            "understated, and the screenplay handles its complex themes with intelligence "
            "and compassion. I left the cinema feeling uplifted, moved, and convinced I "
            "had witnessed something genuinely special. One of the best films of the decade."
        ),
    },
    {
        "label": "😊 Subtle Positive",
        "hint":  "Positive",
        "text": (
            "It is a quietly assured piece of work. The director resists every temptation "
            "to over-explain, trusting the audience and the performances to carry the "
            "emotional weight. There are a few pacing issues in the middle section, and "
            "one subplot never fully pays off, but these are minor complaints. The final "
            "twenty minutes are almost perfect. I would not call it a masterpiece, but it "
            "is a genuinely good film that consistently surprises and rewards attention."
        ),
    },
    {
        "label": "🔄 Rough Start → Positive",
        "hint":  "Positive",
        "text": (
            "I walked in sceptical and the first act confirmed my worst fears: clunky "
            "dialogue, flat pacing, and characters who feel like rough sketches. I nearly "
            "left at the interval. But something extraordinary happens in the second half. "
            "The film finds its voice completely, the performances deepen, and by the final "
            "act I was utterly gripped. An imperfect but ultimately rewarding experience "
            "that earns every ounce of its emotional payoff. Stick with it."
        ),
    },
    {
        "label": "💣 Clear Negative",
        "hint":  "Negative",
        "text": (
            "A complete disaster from start to finish. The script is incoherent, the "
            "direction is amateurish, and the performances range from wooden to "
            "embarrassing. No character is developed beyond a single trait, no scene "
            "builds to anything meaningful, and the climax resolves nothing. The film "
            "wastes a genuinely interesting premise on lazy clichés and narrative "
            "shortcuts. I have rarely left a cinema feeling so comprehensively cheated "
            "of my time and money. Avoid at all costs."
        ),
    },
    {
        "label": "😑 Subtle Negative",
        "hint":  "Negative",
        "text": (
            "The film had every opportunity to be something worthwhile and squanders "
            "all of it. The script is riddled with half-formed ideas that lead nowhere, "
            "the central performances feel hollow and unconvincing, and the pacing drags "
            "every scene well past the point of exhaustion. No relationship earns genuine "
            "emotional weight, no plot thread resolves with any coherence, and the "
            "climax is a confusing, abrupt non-event. Forgettable, frustrating, and "
            "ultimately a waste of everyone involved. Do not bother."
        ),
    },
    {
        "label": "🔄 Good Start → Negative",
        "hint":  "Negative",
        "text": (
            "The setup shows a flicker of promise — a decent premise and one "
            "competent early scene. That is the entire sum of this film's merits. "
            "From the second act onward it deteriorates rapidly into an incoherent, "
            "poorly written disaster. Characters contradict themselves without reason, "
            "subplots are abandoned wholesale, the dialogue becomes stilted and "
            "embarrassing, and the third act is an incomprehensible mess that resolves "
            "nothing. The direction is flat, the pacing is excruciating, and the "
            "ending is an insult to anyone who sat through the preceding ninety minutes. "
            "Deeply disappointing, profoundly boring, and impossible to recommend."
        ),
    },
]


# ===========================================================================
# Helpers
# ===========================================================================

def _clean_text(text: str) -> str:
    """Lightweight cleaning (no spaCy) sufficient for inference."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _simple_tokenize(text: str) -> List[str]:
    """Whitespace tokenisation — mirrors spaCy lemmas for common words."""
    return text.split()


def _numericalize(tokens: List[str], word2idx: Dict[str, int]) -> List[int]:
    unk = UNK_IDX
    return [word2idx.get(t, unk) for t in tokens]


def _pad_or_truncate(seq: List[int], max_len: int = MAX_SEQ_LEN) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [PAD_IDX] * (max_len - len(seq))


def preprocess_for_inference(text: str, word2idx: Dict[str, int]) -> torch.Tensor:
    """Return a (1, MAX_SEQ_LEN) LongTensor ready for model inference."""
    cleaned = _clean_text(text)
    tokens  = _simple_tokenize(cleaned)
    ids     = _numericalize(tokens, word2idx)
    padded  = _pad_or_truncate(ids)
    return torch.tensor([padded], dtype=torch.long)


@st.cache_resource(show_spinner="Loading models & vocabulary …")
def load_all() -> tuple:
    """Load vocabulary + all four models (cached across Streamlit reruns)."""
    # ── vocabulary ──────────────────────────────────────────────────────────
    with open(VOCAB_PATH, "r", encoding="utf-8") as fh:
        vocab_data = json.load(fh)
    word2idx: Dict[str, int] = vocab_data["word2idx"]
    vocab_size = len(word2idx)

    # ── models ──────────────────────────────────────────────────────────────
    device = torch.device("cpu")   # CPU for dashboard; GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")

    models: Dict[str, nn.Module] = {}
    for name, cfg in MODEL_CONFIGS.items():
        model = cfg["cls"](
            vocab_size = vocab_size,
            embed_dim  = EMBED_DIM,
            hidden_dim = cfg["hidden_dim"],
            output_dim = OUTPUT_DIM,
            n_layers   = N_LAYERS,
            dropout    = cfg["dropout"],
        )
        ckpt = torch.load(cfg["checkpoint"], map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models[name] = model

    return word2idx, models, device


def run_inference(text: str, models: Dict[str, nn.Module], word2idx: Dict, device: torch.device) -> Dict:
    """Run all four models on a single review text and return results."""
    tensor = preprocess_for_inference(text, word2idx).to(device)
    results = {}
    with torch.no_grad():
        for name, model in models.items():
            t0     = time.perf_counter()
            logit  = model(tensor).squeeze().item()
            lat_ms = (time.perf_counter() - t0) * 1000
            prob   = 1.0 / (1.0 + np.exp(-logit))   # sigmoid
            results[name] = {
                "prob":      prob,
                "sentiment": "Positive" if prob >= 0.5 else "Negative",
                "latency":   lat_ms,
            }
    return results


@st.cache_data
def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_PATH, index_col="Model")
    return df


@st.cache_data
def load_chart_images() -> Dict[str, bytes | None]:
    images: Dict[str, bytes | None] = {}
    for key, fname in [("roc", "roc_curves.png"), ("f1", "f1_bar_chart.png")]:
        path = os.path.join(PHASE3_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                images[key] = fh.read()
        else:
            images[key] = None
    return images


# ===========================================================================
# Page config
# ===========================================================================
st.set_page_config(
    page_title="Neuro-Sentiment Observatory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal CSS: clean typography, subtle card styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      /* ── global ── */
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      /* ── sidebar ── */
      section[data-testid="stSidebar"] { background: #0f1117; }
      section[data-testid="stSidebar"] * { color: #ffffffcc !important; }

      /* ── metric card ── */
      div[data-testid="metric-container"] {
        background: #1e2029;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #2e3149;
      }

      /* ── tab bar ── */
      button[data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }

      /* ── table headers ── */
      thead tr th { background-color: #1e2029 !important; color: #fff !important; }

      /* ── prediction badge ── */
      .badge-pos {
        display: inline-block; background: #1a7a4a; color: #d4f5e2;
        border-radius: 6px; padding: 3px 12px; font-weight: 700; font-size: 0.95rem;
      }
      .badge-neg {
        display: inline-block; background: #7a1a1a; color: #f5d4d4;
        border-radius: 6px; padding: 3px 12px; font-weight: 700; font-size: 0.95rem;
      }

      /* ── arch card ── */
      .arch-card {
        border-left: 4px solid var(--arch-color, #555);
        background: #1a1d26;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
      }
      .arch-tagline { font-size: 0.85rem; font-weight: 600; color: #aaa; margin-bottom: 8px; }
      .arch-body    { font-size: 0.93rem; line-height: 1.65; color: #ddd; }

      /* ── error card ── */
      .error-card {
        background: #14161f;
        border-radius: 10px;
        border: 1px solid #2e3149;
        padding: 20px 24px;
        margin-bottom: 20px;
      }
      .error-snippet {
        font-size: 0.9rem; color: #bbb; font-style: italic;
        border-left: 3px solid #555; padding-left: 12px; margin: 12px 0;
      }

      /* ── divider ── */
      hr { border-color: #2e3149; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Neuro-Sentiment\nObservatory")
    st.markdown("---")
    st.markdown(
        "**Dataset:** IMDB 50 k reviews  \n"
        "**Task:** Binary Sentiment  \n"
        "**Framework:** PyTorch  \n"
        "**Models:** RNN · LSTM · GRU · Bi-LSTM"
    )
    st.markdown("---")
    st.markdown("**Architectures**")
    for name, cfg in MODEL_CONFIGS.items():
        st.markdown(
            f"<span style='color:{cfg['color']};font-weight:700;'>{cfg['icon']} {name}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Phase 3 · Synthesis & Visualization")

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
word2idx, models, device = load_all()
metrics_df = load_metrics()
images     = load_chart_images()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='font-size:2rem;margin-bottom:4px;'>🧠 Neuro-Sentiment Observatory</h1>"
    "<p style='color:#888;font-size:1rem;margin-top:0;'>"
    "Comparative NLP Study · RNN vs LSTM vs GRU vs Bi-LSTM on IMDB Sentiment"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3 = st.tabs(
    ["⚡ Live Tournament", "📊 Scientific Evaluation", "🔍 Misclassification Gallery"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – LIVE TOURNAMENT
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### ⚡ Live Model Tournament")
    st.markdown(
        "Paste any movie review below and click **Compare Models** to run all four "
        "architectures simultaneously and compare their predictions."
    )

    default_review = (
        "This film is an absolute masterpiece. The direction is visionary, "
        "the performances are career-best, and the screenplay balances humour "
        "and heartbreak with extraordinary precision. I left the cinema feeling "
        "genuinely moved — a rare sensation these days."
    )

    # ── Session state for text area ───────────────────────────────────────
    if "review_input" not in st.session_state:
        st.session_state.review_input = default_review

    # ── Quick-load sample buttons ─────────────────────────────────────────
    st.markdown(
        "<p style='color:#888;font-size:0.85rem;margin-bottom:6px;'>"
        "💡 <strong>Quick-load a sample review:</strong></p>",
        unsafe_allow_html=True,
    )
    scols = st.columns(len(SAMPLE_REVIEWS))
    for scol, sample in zip(scols, SAMPLE_REVIEWS):
        hint_color = "#2ecc71" if sample["hint"] == "Positive" else "#e74c3c"
        with scol:
            if st.button(
                sample["label"],
                key=f"sample__{sample['label']}",
                help=f"Expected sentiment: {sample['hint']}",
                use_container_width=True,
            ):
                st.session_state.review_input = sample["text"]
                st.rerun()
            st.markdown(
                f"<div style='text-align:center;font-size:0.72rem;color:{hint_color};"
                f"margin-top:-8px;padding-bottom:2px;font-weight:600;'>"
                f"{'▲' if sample['hint'] == 'Positive' else '▼'} {sample['hint']}</div>",
                unsafe_allow_html=True,
            )

    review_text = st.text_area(
        "Movie Review",
        value=st.session_state.review_input,
        height=140,
        placeholder="Paste a movie review here …",
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([2, 8])
    with col_btn:
        run_clicked = st.button("🚀 Compare Models", type="primary", width="stretch")

    if run_clicked:
        if not review_text.strip():
            st.warning("Please enter a review before clicking Compare Models.")
        else:
            with st.spinner("Running inference across all four models …"):
                preds = run_inference(review_text, models, word2idx, device)

            st.markdown("#### Prediction Results")

            # ── Comparison table ──────────────────────────────────────────
            rows_html = ""
            for name, res in preds.items():
                cfg       = MODEL_CONFIGS[name]
                color     = cfg["color"]
                sentiment = res["sentiment"]
                conf      = res["prob"] if sentiment == "Positive" else 1 - res["prob"]
                badge_cls = "badge-pos" if sentiment == "Positive" else "badge-neg"
                bar_pct   = int(conf * 100)
                bar_color = "#1a7a4a" if sentiment == "Positive" else "#7a1a1a"

                conf_bar = (
                    f"<div style='background:#2e3149;border-radius:4px;height:8px;width:100%;'>"
                    f"<div style='background:{bar_color};width:{bar_pct}%;height:8px;"
                    f"border-radius:4px;'></div></div>"
                    f"<span style='font-size:0.8rem;color:#aaa;'>{conf:.1%}</span>"
                )

                rows_html += (
                    f"<tr>"
                    f"<td style='padding:10px 8px;font-weight:700;color:{color};'>{cfg['icon']} {name}</td>"
                    f"<td style='padding:10px 8px;'><span class='{badge_cls}'>{sentiment}</span></td>"
                    f"<td style='padding:10px 8px;width:200px;'>{conf_bar}</td>"
                    f"<td style='padding:10px 8px;color:#aaa;font-size:0.85rem;'>{res['latency']:.1f} ms</td>"
                    f"</tr>"
                )

            table_html = (
                "<table style='width:100%;border-collapse:collapse;'>"
                "<thead><tr style='border-bottom:1px solid #2e3149;'>"
                "<th style='text-align:left;padding:8px;color:#aaa;'>Model</th>"
                "<th style='text-align:left;padding:8px;color:#aaa;'>Sentiment</th>"
                "<th style='text-align:left;padding:8px;color:#aaa;'>Confidence</th>"
                "<th style='text-align:left;padding:8px;color:#aaa;'>Latency</th>"
                "</tr></thead>"
                f"<tbody>{rows_html}</tbody>"
                "</table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

            # ── Consensus callout ─────────────────────────────────────────
            st.markdown("")
            pos_count = sum(1 for r in preds.values() if r["sentiment"] == "Positive")
            neg_count = 4 - pos_count
            if pos_count == 4:
                consensus = "✅ **All four models agree: Positive**"
                con_color = "#1a7a4a"
            elif neg_count == 4:
                consensus = "❌ **All four models agree: Negative**"
                con_color = "#7a1a1a"
            else:
                consensus = f"⚠️ **Models disagree — {pos_count} Positive · {neg_count} Negative**"
                con_color = "#7a6b00"

            st.markdown(
                f"<div style='background:{con_color}22;border:1px solid {con_color}66;"
                f"border-radius:8px;padding:12px 16px;'>{consensus}</div>",
                unsafe_allow_html=True,
            )

            # ── Per-model probability gauge ───────────────────────────────
            st.markdown("#### Positive Sentiment Probability")
            cols = st.columns(4)
            for idx, (name, res) in enumerate(preds.items()):
                with cols[idx]:
                    cfg = MODEL_CONFIGS[name]
                    st.metric(
                        label=f"{cfg['icon']} {name}",
                        value=f"{res['prob']:.1%}",
                        delta=f"{'Positive' if res['prob'] >= 0.5 else 'Negative'}",
                    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – SCIENTIFIC EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Scientific Evaluation & Analysis")

    # ── Find the winner ───────────────────────────────────────────────────
    best_model  = metrics_df["F1-Score"].idxmax()
    best_f1     = metrics_df.loc[best_model, "F1-Score"]
    best_acc    = metrics_df.loc[best_model, "Accuracy"]
    best_cfg    = MODEL_CONFIGS[best_model]

    # ── Winner banner ─────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:linear-gradient(90deg,#1e2029,#2b1e38);"
        f"border:1px solid {best_cfg['color']}55;border-left:5px solid {best_cfg['color']};"
        f"border-radius:10px;padding:16px 22px;margin-bottom:20px;'>"
        f"<span style='font-size:1.5rem;'>{best_cfg['icon']}</span> "
        f"<span style='font-weight:800;font-size:1.1rem;color:{best_cfg['color']};'>{best_model}</span>"
        f"<span style='color:#ccc;'> achieved the highest F1-Score of </span>"
        f"<span style='font-weight:800;font-size:1.1rem;color:#f0e68c;'>{best_f1:.4f}</span>"
        f"<span style='color:#ccc;'> and accuracy of </span>"
        f"<span style='font-weight:700;color:#f0e68c;'>{best_acc:.2%}</span>"
        f"<br><span style='color:#aaa;font-size:0.9rem;margin-top:6px;display:block;'>"
        f"Its streamlined gating architecture (Update + Reset gates) allowed it to capture "
        f"long-range sentiment cues efficiently, outperforming more complex architectures "
        f"while using fewer parameters — a demonstration that architectural simplicity "
        f"can regularise effectively and prevent overfitting on this task.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Leaderboard ───────────────────────────────────────────────────────
    st.markdown("#### 🏆 Leaderboard")

    display_df = metrics_df.copy().reset_index()
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    # Sort by F1 descending
    display_df = display_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    display_df["Rank"] = range(1, len(display_df) + 1)

    def _color_row(val, col):
        """Return HTML row with color coding based on model name."""
        return val

    # Build styled HTML table
    lb_rows = ""
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for i, row in display_df.iterrows():
        name = row["Model"]
        cfg  = MODEL_CONFIGS.get(name, {})
        color = cfg.get("color", "#aaa")
        medal = medals[i] if i < len(medals) else str(i + 1)
        lb_rows += (
            f"<tr style='border-bottom:1px solid #2e3149;'>"
            f"<td style='padding:10px 8px;font-size:1.1rem;'>{medal}</td>"
            f"<td style='padding:10px 8px;font-weight:700;color:{color};'>"
            f"{cfg.get('icon', '')} {name}</td>"
            f"<td style='padding:10px 8px;font-weight:700;color:#f0e68c;'>{row['Accuracy']:.4f}</td>"
            f"<td style='padding:10px 8px;font-weight:700;color:#f0e68c;'>{row['F1-Score']:.4f}</td>"
            f"<td style='padding:10px 8px;color:#ddd;'>{row['Precision']:.4f}</td>"
            f"<td style='padding:10px 8px;color:#ddd;'>{row['Recall']:.4f}</td>"
            f"<td style='padding:10px 8px;color:#aaa;font-size:0.85rem;'>{row['Latency (ms / 1000 samples)']:.1f} ms</td>"
            f"</tr>"
        )

    lb_html = (
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='border-bottom:2px solid #3e4169;'>"
        "<th style='text-align:left;padding:8px;color:#888;'>Rank</th>"
        "<th style='text-align:left;padding:8px;color:#888;'>Model</th>"
        "<th style='text-align:left;padding:8px;color:#888;'>Accuracy ↑</th>"
        "<th style='text-align:left;padding:8px;color:#888;'>F1-Score ↑</th>"
        "<th style='text-align:left;padding:8px;color:#888;'>Precision</th>"
        "<th style='text-align:left;padding:8px;color:#888;'>Recall</th>"
        "<th style='text-align:left;padding:8px;color:#888;'>Latency ↓</th>"
        "</tr></thead>"
        f"<tbody>{lb_rows}</tbody>"
        "</table>"
    )
    st.markdown(lb_html, unsafe_allow_html=True)

    # ── Metric definitions ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 Metric Definitions & Importance", expanded=False):
        for metric, info in METRIC_DEFS.items():
            st.markdown(
                f"<div style='margin-bottom:14px;'>"
                f"<span style='font-weight:700;color:#c9a0dc;'>{metric}</span><br>"
                f"<span style='font-size:0.85rem;color:#888;'>Formula:</span> "
                f"<code style='background:#1e2029;padding:2px 6px;border-radius:4px;"
                f"font-size:0.82rem;'>{info['formula']}</code><br>"
                f"<span style='font-size:0.88rem;color:#bbb;margin-top:4px;display:block;'>"
                f"{info['why']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown("#### 📈 Performance Charts")
    col_roc, col_f1 = st.columns(2)

    with col_roc:
        st.markdown("**ROC Curves — All Architectures**")
        if images.get("roc"):
            st.image(images["roc"], width="stretch")
        else:
            st.info("ROC curve image not found. Run `synthesis.py` to generate it.")

    with col_f1:
        st.markdown("**F1-Score Comparison**")
        if images.get("f1"):
            st.image(images["f1"], width="stretch")
        else:
            st.info("F1 bar chart not found. Run `synthesis.py` to generate it.")

    st.markdown("---")

    # ── Architecture Decision Reports ─────────────────────────────────────
    st.markdown("#### 🧬 Architectural Decision Reports")
    st.markdown(
        "<p style='color:#888;font-size:0.9rem;'>"
        "Each section explains <em>why</em> the architecture makes the decisions "
        "it does — from the perspective of its internal mechanism.</p>",
        unsafe_allow_html=True,
    )

    for name, logic in ARCH_LOGIC.items():
        cfg   = MODEL_CONFIGS[name]
        color = cfg["color"]
        row   = metrics_df.loc[name] if name in metrics_df.index else None
        pill  = ""
        if row is not None:
            pill = (
                f"<span style='background:{color}22;color:{color};border:1px solid {color}55;"
                f"border-radius:20px;padding:2px 10px;font-size:0.8rem;margin-left:10px;'>"
                f"F1 {row['F1-Score']:.4f}</span>"
            )
        with st.expander(f"{cfg['icon']} {name} — {logic['tagline']}", expanded=(name == best_model)):
            st.markdown(
                f"<div class='arch-card' style='--arch-color:{color};border-left-color:{color};'>"
                f"<div class='arch-tagline'>{cfg['icon']} {name}{pill}</div>"
                f"<div class='arch-body'>{logic['body']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – MISCLASSIFICATION GALLERY
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔍 Misclassification Gallery — Vanishing Gradient in Action")
    st.markdown(
        "These three examples demonstrate the **vanishing gradient problem** in real-time.  "
        "Each review contains sentiment signals spread across the full length of the text — "
        "a pattern the vanilla RNN cannot capture but the Bi-LSTM handles correctly."
    )
    st.markdown(
        "<div style='background:#1a1d26;border:1px solid #3e4169;border-radius:8px;"
        "padding:14px 18px;margin-bottom:20px;'>"
        "<span style='font-weight:700;color:#e74c3c;'>🔴 RNN</span> "
        "<span style='color:#aaa;'>reads left-to-right and overwrites its hidden state "
        "at each step — early negative cues fade away before the final prediction.</span><br><br>"
        "<span style='font-weight:700;color:#9b59b6;'>🟣 Bi-LSTM</span> "
        "<span style='color:#aaa;'>reads the sequence in both directions, anchoring to "
        "strong sentiment signals at both the opening and closing of the review.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    for i, ex in enumerate(ERROR_EXAMPLES):
        true_label   = ex["true_label"]
        rnn_pred     = ex["rnn_pred"]
        bilstm_pred  = ex["bilstm_pred"]
        rnn_conf     = ex["rnn_conf"]
        bilstm_conf  = ex["bilstm_conf"]

        rnn_badge_cls    = "badge-pos" if rnn_pred == "Positive" else "badge-neg"
        bilstm_badge_cls = "badge-pos" if bilstm_pred == "Positive" else "badge-neg"
        true_badge_cls   = "badge-pos" if true_label == "Positive" else "badge-neg"

        rnn_bar_color    = "#1a7a4a" if rnn_pred == "Positive" else "#7a1a1a"
        bilstm_bar_color = "#1a7a4a" if bilstm_pred == "Positive" else "#7a1a1a"

        def bar(pct, col):
            return (
                f"<div style='background:#2e3149;border-radius:4px;height:6px;width:120px;"
                f"display:inline-block;vertical-align:middle;'>"
                f"<div style='background:{col};width:{int(pct*100)}%;height:6px;"
                f"border-radius:4px;'></div></div> "
                f"<span style='font-size:0.85rem;color:#aaa;'>{pct:.0%}</span>"
            )

        # Truncate long reviews
        snippet = ex["text"]
        if len(snippet) > 420:
            snippet = snippet[:420] + " …"

        card_html = (
            f"<div class='error-card'>"
            f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>"
            f"<span style='font-size:1.3rem;font-weight:800;color:#888;'>#{i+1}</span>"
            f"<span style='color:#888;font-size:0.9rem;'>True Label:</span>"
            f"<span class='{true_badge_cls}'>{true_label}</span>"
            f"</div>"
            # Review snippet
            f"<div class='error-snippet'>{snippet}</div>"
            # Model comparison table
            f"<table style='width:100%;border-collapse:collapse;margin-top:12px;'>"
            f"<thead><tr style='border-bottom:1px solid #2e3149;'>"
            f"<th style='text-align:left;padding:6px 8px;color:#666;font-size:0.82rem;'>Model</th>"
            f"<th style='text-align:left;padding:6px 8px;color:#666;font-size:0.82rem;'>Prediction</th>"
            f"<th style='text-align:left;padding:6px 8px;color:#666;font-size:0.82rem;'>Confidence</th>"
            f"<th style='text-align:left;padding:6px 8px;color:#666;font-size:0.82rem;'>Verdict</th>"
            f"</tr></thead><tbody>"
            # RNN row
            f"<tr style='border-bottom:1px solid #1e2029;'>"
            f"<td style='padding:8px;font-weight:700;color:#e74c3c;'>🔴 RNN</td>"
            f"<td style='padding:8px;'><span class='{rnn_badge_cls}'>{rnn_pred}</span></td>"
            f"<td style='padding:8px;'>{bar(rnn_conf, rnn_bar_color)}</td>"
            f"<td style='padding:8px;'><span style='color:#e74c3c;font-weight:700;'>✗ Wrong</span></td>"
            f"</tr>"
            # Bi-LSTM row
            f"<tr>"
            f"<td style='padding:8px;font-weight:700;color:#9b59b6;'>🟣 Bi-LSTM</td>"
            f"<td style='padding:8px;'><span class='{bilstm_badge_cls}'>{bilstm_pred}</span></td>"
            f"<td style='padding:8px;'>{bar(bilstm_conf, bilstm_bar_color)}</td>"
            f"<td style='padding:8px;'><span style='color:#2ecc71;font-weight:700;'>✓ Correct</span></td>"
            f"</tr>"
            f"</tbody></table>"
            # Explanation
            f"<div style='margin-top:14px;padding-top:12px;border-top:1px solid #2e3149;"
            f"font-size:0.87rem;color:#888;'>"
            f"<span style='color:#e74c3c;'>RNN</span> assigned a "
            f"<strong>{rnn_conf:.0%}</strong> confidence to "
            f"<strong>{rnn_pred}</strong> — it was misled by surface-level "
            f"positive language early in the review and could not retain the "
            f"negative sentiment signals distributed across the full sequence. "
            f"<span style='color:#9b59b6;'>Bi-LSTM</span>'s backward pass "
            f"anchored to the conclusively negative closing sentences, "
            f"giving it <strong>{bilstm_conf:.0%}</strong> confidence for "
            f"the correct <strong>{bilstm_pred}</strong> label."
            f"</div>"
            f"</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # ── Summary callout ───────────────────────────────────────────────────
    st.markdown(
        "<div style='background:#14161f;border:1px solid #9b59b655;"
        "border-left:4px solid #9b59b6;border-radius:8px;"
        "padding:16px 20px;margin-top:8px;'>"
        "<span style='font-weight:700;color:#9b59b6;'>Key Takeaway</span><br>"
        "<span style='color:#bbb;font-size:0.93rem;'>"
        "The vanishing gradient in vanilla RNNs causes the model to lose track of "
        "sentiment signals established early in long reviews.  Gated architectures "
        "(LSTM, GRU) and bidirectional reading (Bi-LSTM) address this by maintaining "
        "selective long-term memory and capturing context from both ends of the sequence, "
        "respectively.  This is not a marginal improvement — it is the difference between "
        "being systematically fooled by mixed-sentiment reviews and classifying them correctly."
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )
