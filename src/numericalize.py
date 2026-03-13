"""
src/numericalize.py
-------------------
Converts pre-processed token lists into fixed-length integer sequences
suitable for consumption by a PyTorch Embedding layer.

Strategy
--------
* Sequences *longer* than ``MAX_SEQ_LEN`` are truncated from the right
  (we keep the first ``MAX_SEQ_LEN`` tokens).
* Sequences *shorter* than ``MAX_SEQ_LEN`` are post-padded with
  ``PAD_IDX`` (index 0) so that the padding sits at the tail, which
  avoids distorting what the recurrent model "sees last".
"""

import logging
from typing import List

from config.config import MAX_SEQ_LEN, PAD_IDX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def pad_or_truncate(
    sequence: List[int],
    max_len: int = MAX_SEQ_LEN,
    pad_idx: int = PAD_IDX,
) -> List[int]:
    """
    Bring *sequence* to exactly *max_len* elements.

    Parameters
    ----------
    sequence : List[int]
        Integer-encoded token list.
    max_len : int
        Target length.  Defaults to :data:`config.config.MAX_SEQ_LEN`.
    pad_idx : int
        Index used for padding.  Defaults to :data:`config.config.PAD_IDX`.

    Returns
    -------
    List[int]
        A list of exactly *max_len* integers.
    """
    if len(sequence) >= max_len:
        return sequence[:max_len]                              # truncate
    return sequence + [pad_idx] * (max_len - len(sequence))   # post-pad


def numericalize_and_pad(
    tokens: List[str],
    vocab,
    max_len: int = MAX_SEQ_LEN,
) -> List[int]:
    """
    Full single-sample pipeline: tokens → integer indices → fixed length.

    Parameters
    ----------
    tokens : List[str]
        Pre-processed token list (output of preprocessing pipeline).
    vocab : Vocabulary
        Fitted :class:`src.vocabulary.Vocabulary` instance.
    max_len : int
        Target sequence length.

    Returns
    -------
    List[int]
        Padded / truncated integer sequence of length *max_len*.
    """
    indices = vocab.encode(tokens)
    return pad_or_truncate(indices, max_len)


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def numericalize_series(
    tokenized_texts: List[List[str]],
    vocab,
    max_len: int = MAX_SEQ_LEN,
) -> List[List[int]]:
    """
    Numericalize and pad an entire split (train / val / test).

    Parameters
    ----------
    tokenized_texts : List[List[str]]
        Output of :func:`src.preprocessing.preprocess_series`.
    vocab : Vocabulary
        **Already fitted** vocabulary (trained on the training split).
    max_len : int
        Target sequence length.

    Returns
    -------
    List[List[int]]
        One padded integer sequence per document.

    Leakage note
    ------------
    This function only *encodes* — it never updates the vocabulary,
    so it is safe to call on validation and test splits.
    """
    sequences = [numericalize_and_pad(tokens, vocab, max_len) for tokens in tokenized_texts]
    logger.info(
        "Numericalized %d sequences  →  each of length %d",
        len(sequences),
        max_len,
    )
    return sequences
