"""
models/gru_model.py
-------------------
Gated Recurrent Unit (GRU) network for binary sentiment classification.

Architecture
------------
  Embedding  →  nn.GRU (stacked, unidirectional)  →  Dropout  →  Linear

Best hyperparameters (avg_val_f1 = 0.8774 ± 0.0047)
-----------------------------------------------------
  hidden_dim : 256
  dropout    : 0.5
  lr         : 0.001
  optimizer  : adam

GRU vs LSTM
-----------
GRUs use only two gates (reset and update) instead of three, making them
~33 % cheaper per step while achieving comparable empirical accuracy on
many NLP tasks.  They are also easier to tune because there are fewer
hyperparameters.

Weight initialisation
---------------------
* Input-hidden weights  : Xavier uniform
* Hidden-hidden weights : Orthogonal
* Biases                : Zeros
* Embedding / FC        : Xavier uniform
"""

import torch
import torch.nn as nn

from config.config import PAD_IDX


class GRUClassifier(nn.Module):
    """
    Stacked unidirectional GRU for binary sentiment classification.

    Parameters
    ----------
    vocab_size  : int   – vocabulary size
    embed_dim   : int   – embedding dimensionality
    hidden_dim  : int   – GRU hidden-state size
    output_dim  : int   – output logits (1 for binary)
    n_layers    : int   – number of stacked GRU layers
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
        # 2. Recurrent engine – GRU
        # ------------------------------------------------------------------
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # ------------------------------------------------------------------
        # 3. Classifier head
        # ------------------------------------------------------------------
        self.fc      = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # 4. Weight initialisation
        # ------------------------------------------------------------------
        self._init_weights()

    # ----------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier / orthogonal initialisation."""
        # Embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(
                self.embedding.weight[self.embedding.padding_idx]
            )

        # GRU internal weights
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

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

        # output → (batch, seq_len, hidden_dim)
        # hidden → (n_layers, batch, hidden_dim)
        _, hidden = self.gru(embedded)

        # Take the last stacked layer's hidden state
        sentence_repr = self.dropout(hidden[-1])   # (batch, hidden_dim)

        # → (batch, output_dim)
        return self.fc(sentence_repr)
