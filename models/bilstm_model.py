"""
models/bilstm_model.py
----------------------
Bidirectional LSTM for binary sentiment classification.

Architecture
------------
  Embedding  →  nn.LSTM(bidirectional=True, stacked)  →  Dropout  →  Linear

Why Bidirectional?
------------------
A standard LSTM reads the sequence left-to-right, so the final hidden state
has "seen" all past tokens but nothing ahead.  A Bi-LSTM runs two separate
LSTM chains in parallel — one forward and one backward — and **concatenates**
their final hidden states.  This gives the classifier access to both left
and right context for every position simultaneously, which typically
outperforms a unidirectional LSTM on classification tasks.

Dimension note
--------------
Because two directions are concatenated the FC input is
``hidden_dim * 2``, not ``hidden_dim``.  PyTorch stores the directions
in the ``hidden`` tensor at positions ``[-2]`` (last forward layer) and
``[-1]`` (last backward layer) when ``n_layers > 1``:

    hidden → (n_layers * 2, batch, hidden_dim)
    final  = cat(hidden[-2], hidden[-1])  →  (batch, hidden_dim * 2)

Weight initialisation
---------------------
* Input-hidden weights  : Xavier uniform
* Hidden-hidden weights : Orthogonal
* Biases                : Zeros; forget-gate bias = 1
* Embedding / FC        : Xavier uniform
"""

import torch
import torch.nn as nn

from config.config import PAD_IDX


class BiLSTMClassifier(nn.Module):
    """
    Stacked Bidirectional LSTM for binary sentiment classification.

    Parameters
    ----------
    vocab_size  : int   – vocabulary size
    embed_dim   : int   – embedding dimensionality
    hidden_dim  : int   – hidden-state size *per direction*
    output_dim  : int   – output logits (1 for binary)
    n_layers    : int   – number of stacked BiLSTM layers
    dropout     : float – dropout probability
    pad_idx     : int   – <PAD> token index
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        output_dim: int,
        n_layers:   int,
        dropout:    float,
        pad_idx:    int = PAD_IDX,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Embedding layer
        # ------------------------------------------------------------------
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        # ------------------------------------------------------------------
        # 2. Recurrent engine – Bidirectional LSTM
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,         # ← the key difference
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # ------------------------------------------------------------------
        # 3. Classifier head
        #    FC input = hidden_dim * 2  (forward + backward concatenated)
        # ------------------------------------------------------------------
        self.fc      = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # 4. Weight initialisation
        # ------------------------------------------------------------------
        self._init_weights()

    # ----------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier / orthogonal initialisation with forget-gate bias = 1."""
        # Embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(
                self.embedding.weight[self.embedding.padding_idx]
            )

        # BiLSTM internal weights
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget-gate bias to 1 for both directions
                hidden_size = param.shape[0] // 4
                param.data[hidden_size : 2 * hidden_size].fill_(1.0)

        # Fully-connected head
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor – (batch_size, seq_len), dtype long

        Returns
        -------
        torch.Tensor – (batch_size, output_dim), raw logits
        """
        # → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(text))

        # output → (batch, seq_len, hidden_dim * 2)
        # hidden → (n_layers * 2, batch, hidden_dim)
        _, (hidden, _cell) = self.lstm(embedded)

        # Concatenate the final forward & backward hidden states
        # hidden[-2] = final forward layer, hidden[-1] = final backward layer
        sentence_repr = self.dropout(
            torch.cat((hidden[-2], hidden[-1]), dim=1)
        )   # → (batch, hidden_dim * 2)

        # → (batch, output_dim)
        return self.fc(sentence_repr)
