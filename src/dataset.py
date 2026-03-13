"""
src/dataset.py
--------------
PyTorch Dataset wrapper and DataLoader factory for the IMDb dataset.

IMDbDataset
-----------
A map-style :class:`torch.utils.data.Dataset` that holds a tensor of
integer sequences and a tensor of binary labels.

create_dataloaders
------------------
Convenience factory that returns three :class:`~torch.utils.data.DataLoader`
objects (train, val, test) with appropriate shuffle settings.
"""

import logging
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config.config import BATCH_SIZE, NUM_WORKERS, PAD_IDX

logger = logging.getLogger(__name__)


class IMDbDataset(Dataset):
    """
    Wraps numericalized, padded integer sequences and binary labels.

    Parameters
    ----------
    sequences : List[List[int]]
        Each inner list is a padded integer sequence of fixed length
        (output of :func:`src.numericalize.numericalize_series`).
    labels : List[int]
        Binary integer labels: 0 = negative, 1 = positive.

    Notes
    -----
    Tensors are created once in ``__init__`` and stored on CPU memory.
    The DataLoader worker processes will handle moving batches to the
    target device via a collate function or inside the training loop.
    """

    def __init__(self, sequences: List[List[int]], labels: List[int]) -> None:
        if len(sequences) != len(labels):
            raise ValueError(
                f"sequences and labels must have the same length, "
                f"got {len(sequences)} and {len(labels)}"
            )

        # dtype=torch.long is required by nn.Embedding
        self.sequences: torch.Tensor = torch.tensor(sequences, dtype=torch.long)
        self.labels:    torch.Tensor = torch.tensor(labels,    dtype=torch.long)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        (sequence, label) : Tuple[torch.Tensor, torch.Tensor]
            sequence – shape (MAX_SEQ_LEN,), dtype torch.long
            label    – scalar, dtype torch.long
        """
        return self.sequences[idx], self.labels[idx]

    def __repr__(self) -> str:
        return (
            f"IMDbDataset(n_samples={len(self)}, "
            f"seq_len={self.sequences.shape[1]})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    train_sequences: List[List[int]],
    train_labels:    List[int],
    val_sequences:   List[List[int]],
    val_labels:      List[int],
    test_sequences:  List[List[int]],
    test_labels:     List[int],
    batch_size:      int = BATCH_SIZE,
    num_workers:     int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build :class:`IMDbDataset` objects and wrap them in DataLoaders.

    Parameters
    ----------
    *_sequences : List[List[int]]
        Numericalized, padded sequence lists for each split.
    *_labels : List[int]
        Integer label lists for each split.
    batch_size : int
        Number of samples per batch.  Defaults to
        :data:`config.config.BATCH_SIZE`.
    num_workers : int
        Subprocesses for data loading.  Set to 0 on Windows.

    Returns
    -------
    train_loader, val_loader, test_loader : Tuple[DataLoader, ...]
        * ``train_loader`` – shuffled, to break temporal / index order.
        * ``val_loader``   – not shuffled, for deterministic evaluation.
        * ``test_loader``  – not shuffled, for deterministic evaluation.
    """
    train_dataset = IMDbDataset(train_sequences, train_labels)
    val_dataset   = IMDbDataset(val_sequences,   val_labels)
    test_dataset  = IMDbDataset(test_sequences,  test_labels)

    # Common DataLoader kwargs
    common = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),   # speed up GPU transfer
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **common)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **common)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **common)

    logger.info(
        "DataLoaders created  –  "
        "Train batches: %d | Val batches: %d | Test batches: %d",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader
