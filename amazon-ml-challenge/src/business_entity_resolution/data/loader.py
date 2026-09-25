"""
Data loading module for reading tab-separated business entity datasets.
"""

from typing import Dict, Tuple
import pandas as pd
from pathlib import Path

from ..config import (
    TRAIN_S1_PATH,
    TRAIN_S2_PATH,
    TRAIN_S3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TEST_S1_PATH,
    TEST_S2_PATH,
    TEST_S3_PATH,
    GROUND_TRUTH_S1_COL,
    GROUND_TRUTH_MATCHED_COL,
)


def load_source_tsv(file_path: Path) -> pd.DataFrame:
    """Read a single entity source TSV file with explicit tab separator."""
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    return df


def load_ground_truth(file_path: Path = TRAIN_GROUND_TRUTH_PATH) -> pd.DataFrame:
    """Read ground truth TSV mapping source 1 entity IDs to matched IDs."""
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    return df


def load_all_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training datasets (source 1, source 2, source 3, ground truth)."""
    df_s1 = load_source_tsv(TRAIN_S1_PATH)
    df_s2 = load_source_tsv(TRAIN_S2_PATH)
    df_s3 = load_source_tsv(TRAIN_S3_PATH)
    gt = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
    return df_s1, df_s2, df_s3, gt


def load_all_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test datasets (source 1, source 2, source 3)."""
    df_s1 = load_source_tsv(TEST_S1_PATH)
    df_s2 = load_source_tsv(TEST_S2_PATH)
    df_s3 = load_source_tsv(TEST_S3_PATH)
    return df_s1, df_s2, df_s3
