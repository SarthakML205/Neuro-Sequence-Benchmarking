"""
src/preprocessing.py
--------------------
Text preprocessing pipeline:

  clean_text              – HTML removal, lower-casing, non-alpha stripping
  tokenize_and_lemmatize  – spaCy tokenisation + lemmatisation
  preprocess_series       – applies both steps to an entire pandas Series
                            with an optional tqdm progress bar
"""

import re
import logging
from typing import List

import pandas as pd
import spacy
from tqdm import tqdm

from config.config import TEXT_COL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load spaCy model once at import time.
# Disable components that are not needed to speed up batch processing.
# ---------------------------------------------------------------------------
try:
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError as exc:
    raise OSError(
        "spaCy model 'en_core_web_sm' is not installed.\n"
        "Run:  python -m spacy download en_core_web_sm"
    ) from exc


# ---------------------------------------------------------------------------
# Step 1 – Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Apply rule-based cleaning to a raw review string.

    Steps (in order)
    ----------------
    1. Remove HTML tags  (e.g. ``<br />``, ``<i>``).
    2. Convert to lower-case.
    3. Strip every character that is not a-z or a space.
    4. Collapse consecutive whitespace to a single space.

    Parameters
    ----------
    text : str
        A raw review string, possibly containing HTML entities.

    Returns
    -------
    str
        The cleaned, lower-cased, alpha-only text.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # 2. Lower-case
    text = text.lower()
    # 3. Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)
    # 4. Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Step 2 – Tokenisation & Lemmatisation
# ---------------------------------------------------------------------------

def tokenize_and_lemmatize(text: str) -> List[str]:
    """
    Tokenise *text* with spaCy and return a list of lemmas.

    Whitespace-only tokens are discarded.  All other tokens
    (including stop-words) are kept because the RNN/LSTM/GRU models
    may exploit sequential context.

    Parameters
    ----------
    text : str
        Pre-cleaned text (output of :func:`clean_text`).

    Returns
    -------
    List[str]
        Lemmatised tokens.
    """
    doc = _nlp(text)
    return [token.lemma_ for token in doc if not token.is_space]


# ---------------------------------------------------------------------------
# Step 3 – Batch preprocessing for a pandas Series
# ---------------------------------------------------------------------------

def preprocess_series(
    series: pd.Series,
    batch_size: int = 512,
    show_progress: bool = True,
) -> List[List[str]]:
    """
    Run the full preprocessing pipeline (clean → tokenise → lemmatise)
    on every element of a pandas Series.

    spaCy's :func:`nlp.pipe` is used for efficient batched processing,
    which is significantly faster than calling :func:`nlp` on each text
    individually.

    Parameters
    ----------
    series : pd.Series
        Series of raw review strings.
    batch_size : int
        Number of documents to feed to :func:`nlp.pipe` at once.
    show_progress : bool
        When ``True`` a tqdm progress bar is displayed.

    Returns
    -------
    List[List[str]]
        One list of lemmatised tokens per input text.
    """
    # Clean all texts first (fast, pure-Python step)
    cleaned: List[str] = [clean_text(t) for t in series]

    # spaCy batch processing
    tokenized: List[List[str]] = []
    docs = _nlp.pipe(cleaned, batch_size=batch_size)

    if show_progress:
        docs = tqdm(docs, total=len(cleaned), desc="  Tokenising & Lemmatising")

    for doc in docs:
        tokenized.append([token.lemma_ for token in doc if not token.is_space])

    return tokenized
