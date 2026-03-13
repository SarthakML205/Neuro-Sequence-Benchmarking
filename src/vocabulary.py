"""
src/vocabulary.py
-----------------
Vocabulary class that builds a word ↔ index mapping from the *training*
set only, thereby preventing any leakage of validation / test statistics
into the model's input space.

Special tokens
--------------
<PAD>  (index 0)  –  used for post-padding sequences to a fixed length.
<UNK>  (index 1)  –  replaces any word that is absent from the vocabulary.
"""

import json
import logging
import os
from collections import Counter
from typing import Dict, List, Optional

from config.config import (
    MAX_VOCAB_SIZE,
    OUTPUTS_DIR,
    PAD_IDX,
    PAD_TOKEN,
    UNK_IDX,
    UNK_TOKEN,
)

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Maps tokens to integer indices and back.

    Parameters
    ----------
    max_size : int
        Maximum vocabulary size, including the two special tokens.
        Defaults to :data:`config.config.MAX_VOCAB_SIZE`.
    """

    def __init__(self, max_size: int = MAX_VOCAB_SIZE) -> None:
        self.max_size: int = max_size

        # Mappings (populated by :meth:`build`)
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        # Full frequency counter (useful for diagnostics)
        self.word_counts: Counter = Counter()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, tokenized_texts: List[List[str]]) -> None:
        """
        Construct the vocabulary from a list of token lists.

        **Must only be called on the training split.**

        Algorithm
        ---------
        1. Count every token across all training documents.
        2. Keep the ``max_size - 2`` most frequent tokens
           (leaving slots 0 and 1 for the special tokens).
        3. Assign index 0 → ``<PAD>`` and 1 → ``<UNK>``, then
           assign subsequent indices in descending frequency order.

        Parameters
        ----------
        tokenized_texts : List[List[str]]
            Pre-processed (cleaned + lemmatised) token lists from the
            **training set only**.
        """
        logger.info("Building vocabulary from %d documents …", len(tokenized_texts))

        # Count tokens across the entire training corpus
        for tokens in tokenized_texts:
            self.word_counts.update(tokens)

        logger.info("Total unique tokens in training corpus: %d", len(self.word_counts))

        # Select the most frequent (max_size - 2) words
        top_words = self.word_counts.most_common(self.max_size - 2)

        # Initialise with special tokens
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

        for idx, (word, _freq) in enumerate(top_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        logger.info(
            "Vocabulary built  –  size: %d  (max allowed: %d)",
            len(self.word2idx),
            self.max_size,
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to a list of integer indices.

        Tokens absent from the vocabulary are mapped to ``UNK_IDX``.

        Parameters
        ----------
        tokens : List[str]

        Returns
        -------
        List[int]
        """
        return [self.word2idx.get(token, UNK_IDX) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Convert a list of integer indices back to tokens.

        Parameters
        ----------
        indices : List[int]

        Returns
        -------
        List[str]
        """
        return [self.idx2word.get(idx, UNK_TOKEN) for idx in indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Serialise the vocabulary to a JSON file.

        Parameters
        ----------
        filepath : str, optional
            Destination path.  Defaults to
            ``<OUTPUTS_DIR>/vocabulary.json``.

        Returns
        -------
        str
            The path where the file was written.
        """
        if filepath is None:
            filepath = os.path.join(OUTPUTS_DIR, "vocabulary.json")

        payload = {
            "max_size": self.max_size,
            "word2idx": self.word2idx,
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info("Vocabulary saved to: %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """
        Restore a :class:`Vocabulary` from a previously saved JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON file produced by :meth:`save`.

        Returns
        -------
        Vocabulary
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        vocab = cls(max_size=payload["max_size"])
        vocab.word2idx = payload["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        logger.info("Vocabulary loaded from: %s  (size: %d)", filepath, len(vocab))
        return vocab

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, token: str) -> bool:
        return token in self.word2idx

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={len(self)}, "
            f"max_size={self.max_size}, "
            f"special_tokens=['{PAD_TOKEN}', '{UNK_TOKEN}'])"
        )
