"""
src/__init__.py
---------------
Marks `src` as a Python package and re-exports the public API for
Phase 1 so that the main script can import everything from one place.
"""

from src.data_loader   import load_data, split_data
from src.preprocessing import clean_text, tokenize_and_lemmatize, preprocess_series
from src.vocabulary    import Vocabulary
from src.numericalize  import numericalize_series
from src.dataset       import IMDbDataset, create_dataloaders

__all__ = [
    "load_data",
    "split_data",
    "clean_text",
    "tokenize_and_lemmatize",
    "preprocess_series",
    "Vocabulary",
    "numericalize_series",
    "IMDbDataset",
    "create_dataloaders",
]
