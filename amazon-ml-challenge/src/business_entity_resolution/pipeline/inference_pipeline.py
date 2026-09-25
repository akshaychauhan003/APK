"""
End-to-end inference pipeline orchestrator producing submission TSV output files.

Handles the full 1.7M-entity test set by processing S1 entities in chunks
to avoid memory exhaustion on a single machine.
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
    GROUND_TRUTH_MATCHED_COL,
)

logger = logging.getLogger(__name__)

# Process test S1 in chunks to avoid OOM on the full 1.7M-row test set.
# Each chunk generates its own candidate pairs independently.
INFERENCE_CHUNK_SIZE = 50_000   # S1 entities per chunk


def run_inference_pipeline(
    output_matching_path: Path = OUTPUT_MATCHING_PATH,
    output_candidate_path: Path = OUTPUT_CANDIDATE_PATH,
    threshold: float = CLASSIFICATION_THRESHOLD,
    chunk_size: int = INFERENCE_CHUNK_SIZE,
):
    """Execute end-to-end inference pipeline and write required submission TSV files."""
    logger.info("1. Loading test datasets...")
    df_s1, df_s2, df_s3 = load_all_test_data()
    logger.info(f"   S1={len(df_s1):,}  S2={len(df_s2):,}  S3={len(df_s3):,}")

    logger.info("2. Preprocessing test records (S2 + S3 only — done once)...")
    df_s2_clean = preprocess_dataset(df_s2)
    df_s3_clean = preprocess_dataset(df_s3)
    df_cand_all = pd.concat([df_s2_clean, df_s3_clean], ignore_index=True)
    df_cand_all = df_cand_all.drop_duplicates(subset=[ENTITY_ID_COL])
    logger.info(f"   Combined candidate pool: {len(df_cand_all):,} records")

    # Preprocess S1 fully first (needed for the candidate TSV with all S1 IDs)
    logger.info("3. Preprocessing S1 test records...")
    df_s1_clean = preprocess_dataset(df_s1)
    all_s1_ids = df_s1_clean[ENTITY_ID_COL]

    total_chunks = (len(df_s1_clean) + chunk_size - 1) // chunk_size
    logger.info(
        f"4. Chunked inference: {len(df_s1_clean):,} S1 entities "
        f"in {total_chunks} chunks of {chunk_size:,}..."
    )

    all_pairs_list = []
    all_matching_list = []

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(df_s1_clean))
        df_s1_chunk = df_s1_clean.iloc[start:end].copy()
        chunk_s1_ids = df_s1_chunk[ENTITY_ID_COL]

        logger.info(
            f"   Chunk {chunk_idx + 1}/{total_chunks}: "
            f"S1 rows {start:,}–{end:,}  ({len(df_s1_chunk):,} entities)"
        )

        # Blocking
        df_pairs_chunk = generate_candidate_pairs(df_s1_chunk, df_s2_clean, df_s3_clean)
        logger.info(f"     → {len(df_pairs_chunk):,} candidate pairs")
        all_pairs_list.append(df_pairs_chunk)

        if df_pairs_chunk.empty:
            # All singletons for this chunk
            chunk_matching = pd.DataFrame({
                GROUND_TRUTH_S1_COL: chunk_s1_ids,
                GROUND_TRUTH_MATCHED_COL: "",
            })
            all_matching_list.append(chunk_matching)
            continue

        # Feature extraction
        df_features_chunk = build_feature_matrix(df_pairs_chunk, df_s1_chunk, df_cand_all)

        # Prediction
        chunk_matching = predict_matches(
            df_features_chunk,
            all_s1_ids=chunk_s1_ids,
            threshold=threshold,
        )
        all_matching_list.append(chunk_matching)

    # ── Combine all chunks ──────────────────────────────────────────────────
    logger.info("5. Combining all chunk results...")

    df_all_pairs = pd.concat(all_pairs_list, ignore_index=True)
    df_matching_results = pd.concat(all_matching_list, ignore_index=True)

    # Ensure every S1 entity has exactly one row (de-dup if any chunk boundary issues)
    df_matching_results = df_matching_results.drop_duplicates(
        subset=[GROUND_TRUTH_S1_COL], keep="first"
    )

    logger.info("6. Formatting and saving candidate_pairs.tsv...")
    df_candidate_tsv = format_candidate_pairs_tsv(df_all_pairs, all_s1_ids)
    df_candidate_tsv.to_csv(output_candidate_path, sep="\t", index=False)
    logger.info(f"   Saved {output_candidate_path}  ({len(df_candidate_tsv):,} rows)")

    logger.info("7. Saving matching_results.tsv...")
    df_matching_results.to_csv(output_matching_path, sep="\t", index=False)
    logger.info(f"   Saved {output_matching_path}  ({len(df_matching_results):,} rows)")

    matched_count = (df_matching_results[GROUND_TRUTH_MATCHED_COL] != "").sum()
    singleton_count = (df_matching_results[GROUND_TRUTH_MATCHED_COL] == "").sum()
    logger.info(
        f"   Summary: {matched_count:,} entities with matches, "
        f"{singleton_count:,} singletons"
    )

    logger.info("Inference pipeline complete!")
    return df_matching_results, df_candidate_tsv
