"""
Validation utilities for dataset schemas and entity structures.
"""

import logging
import pandas as pd
from typing import List

from ..config import ENTITY_ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [ENTITY_ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL]


def validate_dataframe_schema(df: pd.DataFrame, source_prefix: str) -> bool:
    """Validate that the dataset contains expected columns and valid prefixes."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for source {source_prefix}: {missing_cols}")

    if df.empty:
        raise ValueError(f"Dataset for source {source_prefix} is empty!")

    invalid_ids = ~df[ENTITY_ID_COL].str.startswith(source_prefix)
    if invalid_ids.any():
        num_invalid = invalid_ids.sum()
        logger.warning(f"Found {num_invalid} records in source {source_prefix} missing prefix '{source_prefix}'.")

    return True


def validate_entity_ids(entity_ids: List[str], expected_prefix: str) -> bool:
    """Validate list of entity IDs for proper prefix formatting."""
    invalid = [eid for eid in entity_ids if not eid.startswith(expected_prefix)]
    if invalid:
        logger.warning(f"{len(invalid)} entity IDs do not match prefix '{expected_prefix}'. Example: {invalid[:3]}")
        return False
    return True
