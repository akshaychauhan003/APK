"""
End-to-end inference pipeline orchestrator producing submission TSV output files.
"""

import logging
import pandas as pd
from pathlib import Path

from ..data import load_all_test_data
from ..preprocessing import preprocess_dataset
from ..blocking import generate_candidate_pairs
from ..blocking.candidate_generation import format_candidate_pairs_tsv
from ..features import build_feature_matrix
from ..models import predict_matches
from ..config import (
    ENTITY_ID_COL,
    OUTPUT_MATCHING_PATH,
    OUTPUT_CANDIDATE_PATH,
    CLASSIFICATION_THRESHOLD,
    GROUND_TRUTH_S1_COL,
)

logger = logging.getLogger(__name__)


def run_inference_pipeline(
    output_matching_path: Path = OUTPUT_MATCHING_PATH,
    output_candidate_path: Path = OUTPUT_CANDIDATE_PATH,
    threshold: float = CLASSIFICATION_THRESHOLD,
):
    """Execute end-to-end inference pipeline and write required submission TSV files."""
    logger.info("1. Loading test datasets...")
    df_s1, df_s2, df_s3 = load_all_test_data()

    logger.info("2. Preprocessing test records...")
    df_s1_clean = preprocess_dataset(df_s1)
    df_s2_clean = preprocess_dataset(df_s2)
    df_s3_clean = preprocess_dataset(df_s3)

    logger.info("3. Running candidate pair blocking stage...")
    df_pairs = generate_candidate_pairs(df_s1_clean, df_s2_clean, df_s3_clean)
    all_s1_ids = df_s1_clean[ENTITY_ID_COL]

    logger.info("4. Formatting and saving candidate_pairs.tsv...")
    df_candidate_tsv = format_candidate_pairs_tsv(df_pairs, all_s1_ids)
    df_candidate_tsv.to_csv(output_candidate_path, sep="\t", index=False)
    logger.info(f"Saved {output_candidate_path}")

    logger.info("5. Building feature matrix for candidate pairs...")
    df_cand_all = pd.concat([df_s2_clean, df_s3_clean], ignore_index=True)
    df_cand_all = df_cand_all.drop_duplicates(subset=[ENTITY_ID_COL])
    df_features = build_feature_matrix(df_pairs, df_s1_clean, df_cand_all)

    logger.info("6. Running prediction and thresholding...")
    df_matching_results = predict_matches(
        df_features,
        all_s1_ids=all_s1_ids,
        threshold=threshold,
    )

    logger.info("7. Saving matching_results.tsv...")
    df_matching_results.to_csv(output_matching_path, sep="\t", index=False)
    logger.info(f"Saved {output_matching_path}")

    logger.info("Inference pipeline complete!")
    return df_matching_results, df_candidate_tsv
