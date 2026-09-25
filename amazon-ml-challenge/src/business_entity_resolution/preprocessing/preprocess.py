"""
Master preprocessing orchestrator for full entity datasets.
"""

import pandas as pd
from .normalize_names import normalize_business_name
from .normalize_addresses import normalize_address, extract_postal_code
from ..config import NAME_COL, ADDRESS_COL


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned name, cleaned address, and extracted postal code columns to dataset."""
    df_clean = df.copy()

    df_clean["clean_name"] = df_clean[NAME_COL].fillna("").apply(normalize_business_name)
    df_clean["clean_address"] = df_clean[ADDRESS_COL].fillna("").apply(normalize_address)
    df_clean["postal_code"] = df_clean[ADDRESS_COL].fillna("").apply(extract_postal_code)

    return df_clean
