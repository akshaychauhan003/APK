"""
Validation helper for hold-out performance evaluation.
"""

import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split

from .metrics import evaluate_predictions
from ..config import RANDOM_STATE, GROUND_TRUTH_S1_COL, ENTITY_ID_COL


def split_validation_data(
    df_s1: pd.DataFrame,
    df_gt: pd.DataFrame,
    test_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split S1 entities into train and validation sets.

    df_s1 uses ENTITY_ID_COL ('entity_id') as its ID column.
    df_gt uses GROUND_TRUTH_S1_COL ('source1_entity_id') for filtering.
    """
    # S1 source files use 'entity_id', not 'source1_entity_id'
    s1_ids = df_s1[ENTITY_ID_COL].unique()
    train_ids, val_ids = train_test_split(s1_ids, test_size=test_size, random_state=RANDOM_STATE)

    train_s1 = df_s1[df_s1[ENTITY_ID_COL].isin(train_ids)].copy()
    val_s1 = df_s1[df_s1[ENTITY_ID_COL].isin(val_ids)].copy()

    # Ground truth file uses 'source1_entity_id'
    train_gt = df_gt[df_gt[GROUND_TRUTH_S1_COL].isin(train_ids)].copy()
    val_gt = df_gt[df_gt[GROUND_TRUTH_S1_COL].isin(val_ids)].copy()

    return train_s1, val_s1, train_gt, val_gt


def validate_model_performance(df_pred: pd.DataFrame, df_val_gt: pd.DataFrame) -> Dict[str, float]:
    """Run performance evaluation on validation predictions."""
    return evaluate_predictions(df_pred, df_val_gt)
