"""
Feature builder module combining name and address feature extractors.
"""

import pandas as pd
from typing import Dict, List
from tqdm import tqdm

from .name_features import compute_name_features
from .address_features import compute_address_features
from ..config import ENTITY_ID_COL, COUNTRY_COL, GROUND_TRUTH_S1_COL


def build_feature_matrix(
    df_pairs: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix for all candidate pairs in df_pairs."""
    s1_dict = df_s1.set_index(ENTITY_ID_COL).to_dict(orient="index")
    cand_dict = df_candidates.set_index(ENTITY_ID_COL).to_dict(orient="index")

    feature_list: List[Dict[str, float]] = []

    for _, row in df_pairs.iterrows():
        s1_id = row[GROUND_TRUTH_S1_COL]
        cand_id = row["candidate_entity_id"]

        rec1 = s1_dict.get(s1_id, {})
        rec2 = cand_dict.get(cand_id, {})

        name_feats = compute_name_features(
            rec1.get("clean_name", ""),
            rec2.get("clean_name", ""),
        )

        addr_feats = compute_address_features(
            rec1.get("clean_address", ""),
            rec2.get("clean_address", ""),
            rec1.get("postal_code", ""),
            rec2.get("postal_code", ""),
            rec1.get(COUNTRY_COL, ""),
            rec2.get(COUNTRY_COL, ""),
        )

        combined = {
            GROUND_TRUTH_S1_COL: s1_id,
            "candidate_entity_id": cand_id,
            **name_feats,
            **addr_feats,
        }
        feature_list.append(combined)

    df_feats = pd.DataFrame(feature_list)
    return df_feats
