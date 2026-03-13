"""
models/lstm_model.py
--------------------
Long Short-Term Memory (LSTM) network for binary sentiment classification.

Architecture
------------
  Embedding  →  nn.LSTM (stacked, unidirectional)  →  Dropout  →  Linear

Best hyperparameters (avg_val_f1 = 0.8648 ± 0.0047)
-----------------------------------------------------
  hidden_dim : 128
  dropout    : 0.5
  lr         : 0.0001
  optimizer  : adam

Why LSTM over vanilla RNN?
--------------------------
The gated architecture (input, forget, output gates) allows the network to
selectively remember and forget information across long sequences, directly
addressing the vanishing-gradient problem of vanilla RNNs.

Weight initialisation
---------------------
* Input-hidden weights  : Xavier uniform
* Hidden-hidden weights : Orthogonal
* Biases                : Zeros (forget-gate bias optionally set to 1 to
                          encourage remembering early in training)
* Embedding / FC        : Xavier uniform

Notes
-----
``nn.LSTM`` returns ``(output, (hidden, cell))``.
We use the *hidden* state (not *cell*) of the topmost layer as the
sentence representation — hidden encodes the *output* signal of the cell,
which is what the network has decided is relevant right now.
"""

import torch
import torch.nn as nn

from config.config import PAD_IDX


class LSTMClassifier(nn.Module):
    """
    Stacked unidirectional LSTM for binary sentiment classification.

    Parameters
    ----------
    vocab_size  : int   – vocabulary size
    embed_dim   : int   – embedding dimensionality
    hidden_dim  : int   – LSTM hidden-state size
    output_dim  : int   – output logits (1 for binary)
    n_layers    : int   – number of stacked LSTM layers
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
        # 2. Recurrent engine – LSTM
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
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
        """Xavier / orthogonal initialisation with forget-gate bias = 1."""
        # Embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(
                self.embedding.weight[self.embedding.padding_idx]
            )

        # LSTM internal weights
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget-gate bias to 1 to improve long-range memory
                # early in training.  The LSTM bias vector is laid out as
                # [bias_i | bias_f | bias_g | bias_o]; hidden_size = H
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

        # output → (batch, seq_len, hidden_dim)
        # hidden → (n_layers, batch, hidden_dim)
        # cell   → (n_layers, batch, hidden_dim)
        _, (hidden, _cell) = self.lstm(embedded)

        # Use the topmost LSTM layer's hidden state
        # hidden[-1] → (batch, hidden_dim)
        sentence_repr = self.dropout(hidden[-1])

        # → (batch, output_dim)
        return self.fc(sentence_repr)
