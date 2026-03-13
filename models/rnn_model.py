"""
models/rnn_model.py
-------------------
Vanilla Recurrent Neural Network (Elman RNN) for binary sentiment
classification.

Architecture
------------
  Embedding  →  nn.RNN (stacked)  →  Dropout  →  Linear

Best hyperparameters (avg_val_f1 = 0.8486 ± 0.0029)
-----------------------------------------------------
  hidden_dim : 128
  dropout    : 0.3
  lr         : 0.0001
  optimizer  : adam

Weight initialisation
---------------------
* Input-hidden weights  : Xavier uniform   (controls forward signal scale)
* Hidden-hidden weights : Orthogonal       (preserves gradient norms over time)
* Biases                : Zeros
* Embedding             : Xavier uniform
* FC weight             : Xavier uniform
* FC bias               : Zero

Notes
-----
The final *hidden state* of the topmost RNN layer is used as the sentence
representation.  For a 2-layer RNN, ``hidden`` has shape
``(n_layers, batch_size, hidden_dim)``; we take ``hidden[-1]`` (last layer).
Gradient clipping is handled externally in trainer.py.
"""

import torch
import torch.nn as nn

from config.config import PAD_IDX


class RNNClassifier(nn.Module):
    """
    Vanilla (Elman) RNN for binary sentiment classification.

    Parameters
    ----------
    vocab_size  : int   – number of tokens in the vocabulary
    embed_dim   : int   – dimensionality of the token embeddings
    hidden_dim  : int   – number of features in the RNN hidden state
    output_dim  : int   – number of output logits (1 for binary)
    n_layers    : int   – number of stacked RNN layers
    dropout     : float – dropout probability applied to non-recurrent 
                          connections and the classifier head
    pad_idx     : int   – index of the <PAD> token; its embedding is
                          frozen at zero so padding does not contribute
                          to the gradient
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
        # 2. Recurrent engine – vanilla RNN
        # ------------------------------------------------------------------
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            # tanh is the default nonlinearity; relu can also be tried
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
        """Xavier / orthogonal initialisation for stable gradient flow."""
        # Embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        # Zero out the <PAD> row so it stays inert
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(
                self.embedding.weight[self.embedding.padding_idx]
            )

        # RNN internal weights
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:                   # input → hidden
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:                 # hidden → hidden
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
        text : torch.Tensor – shape (batch_size, seq_len), dtype long

        Returns
        -------
        torch.Tensor – shape (batch_size, output_dim), raw logits
        """
        # text → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(text))

        # output → (batch, seq_len, hidden_dim)
        # hidden → (n_layers, batch, hidden_dim)
        _, hidden = self.rnn(embedded)

        # Take the last layer's hidden state as the sentence vector
        # hidden[-1] → (batch, hidden_dim)
        sentence_repr = self.dropout(hidden[-1])

        # → (batch, output_dim)
        return self.fc(sentence_repr)
