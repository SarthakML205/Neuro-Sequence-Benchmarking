"""
src/data_loader.py
------------------
Responsible for:
  1. Reading the raw IMDb CSV from disk.
  2. Encoding string labels to integers.
  3. Splitting the dataset into train / validation / test subsets using
     stratified sampling so that class balance is preserved in every split.

Leakage note
------------
The split is performed *before* any vocabulary construction or
text statistics are computed so that no information from the
validation / test sets bleeds into the training pipeline.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATASET_PATH,
    LABEL_COL,
    LABEL_MAP,
    RANDOM_STATE,
    TEST_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the IMDb CSV and return a cleaned DataFrame.

    Columns returned
    ----------------
    review    : str  – raw review text
    label     : int  – 0 = negative, 1 = positive

    Raises
    ------
    FileNotFoundError  if the CSV is not found at DATASET_PATH.
    ValueError         if unexpected label values are encountered.
    """
    logger.info("Loading dataset from: %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    # Validate expected columns
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Expected column '{LABEL_COL}' in CSV, found: {list(df.columns)}"
        )

    # Encode labels  (positive → 1, negative → 0)
    unknown_labels = set(df[LABEL_COL].unique()) - set(LABEL_MAP.keys())
    if unknown_labels:
        raise ValueError(f"Unexpected label values: {unknown_labels}")

    df["label"] = df[LABEL_COL].map(LABEL_MAP)
    df.drop(columns=[LABEL_COL], inplace=True)

    logger.info("Dataset loaded  – total samples: %d", len(df))
    logger.info(
        "Class distribution:\n%s",
        df["label"].value_counts().to_string(),
    )
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into Train (80 %), Validation (10 %), Test (10 %).

    Parameters
    ----------
    df : pd.DataFrame
        The full, label-encoded dataset returned by :func:`load_data`.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame

    Design
    ------
    * A fixed ``RANDOM_STATE`` guarantees reproducibility.
    * ``stratify=df["label"]`` maintains the original class balance in
      every split, which is critical for fair evaluation.
    * Two-step split avoids rounding artefacts:
        Step 1 – carve out 20 % as *temp*  (val + test combined).
        Step 2 – split *temp* 50 / 50 into val and test.
    """
    temp_size = VAL_RATIO + TEST_RATIO   # 0.20

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Each half of temp_df  →  10 % of the full dataset
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    logger.info(
        "Split sizes  –  Train: %d | Val: %d | Test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True),   \
           test_df.reset_index(drop=True)
