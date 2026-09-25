"""
End-to-end training pipeline orchestrator.

Design for scale:
- S1 is sampled (50k by default) for fast iteration
- S2+S3 preprocessing is done in chunks to handle 10M+ rows
- Blocking uses vectorized pandas merge (not iterrows)
"""

import logging
import pandas as pd
from ..data import load_all_train_data
from ..preprocessing import preprocess_dataset
from ..blocking import generate_candidate_pairs
from ..features import build_feature_matrix
from ..models import train_matching_model
from ..models.predict import predict_matches
from ..evaluation import evaluate_predictions
from ..evaluation.validation import split_validation_data
from ..config import (
    ENTITY_ID_COL,
    GROUND_TRUTH_S1_COL,
    TRAIN_SAMPLE_SIZE,
)

logger = logging.getLogger(__name__)

# For training we sub-sample S2/S3 too — we only need candidates that could
# plausibly match the sampled S1 entities. This dramatically cuts preprocessing time.
# In production inference the full S2/S3 is preprocessed once via inference_pipeline.
CAND_SAMPLE_PER_COUNTRY = 300_000   # Max S2+S3 records per country for training


def _preprocess_candidates_sampled(df_s2: pd.DataFrame, df_s3: pd.DataFrame,
                                   sample_per_country: int) -> pd.DataFrame:
    """
    For training: sample S2+S3 candidates by country to bound preprocessing time.
    Returns a preprocessed combined candidate DataFrame.
    """
    logger.info("  Sub-sampling S2+S3 by country for training...")
    combined = pd.concat([df_s2, df_s3], ignore_index=True)

    from ..config import COUNTRY_COL
    sampled_parts = []
    for country, grp in combined.groupby(COUNTRY_COL):
        if len(grp) > sample_per_country:
            grp = grp.sample(n=sample_per_country, random_state=42)
            logger.info(f"    [{country}]: sampled {sample_per_country:,} from {len(combined[combined[COUNTRY_COL]==country]):,}")
        sampled_parts.append(grp)

    sampled = pd.concat(sampled_parts, ignore_index=True)
    logger.info(f"  Candidate sample size: {len(sampled):,} records → preprocessing...")
    return preprocess_dataset(sampled)


def run_training_pipeline(val_split: float = 0.20, sample_size: int = TRAIN_SAMPLE_SIZE):
    """Execute end-to-end training and validation pipeline.

    Args:
        val_split:   Fraction of S1 entities held out for validation.
        sample_size: Max S1 entities to use for training. Set to None for all.
    """
    logger.info("1. Loading raw training data...")
    df_s1, df_s2, df_s3, df_gt = load_all_train_data()
    logger.info(f"   S1={len(df_s1):,}  S2={len(df_s2):,}  S3={len(df_s3):,}  GT={len(df_gt):,}")

    # ── Sample S1 ─────────────────────────────────────────────────────────
    if sample_size and len(df_s1) > sample_size:
        logger.info(f"   Sampling {sample_size:,} S1 entities from {len(df_s1):,}...")
        df_s1 = df_s1.sample(n=sample_size, random_state=42).reset_index(drop=True)
        sampled_ids = set(df_s1[ENTITY_ID_COL])
        df_gt = df_gt[df_gt[GROUND_TRUTH_S1_COL].isin(sampled_ids)].copy()

    logger.info("2. Preprocessing S1 entities...")
    df_s1_clean = preprocess_dataset(df_s1)

    logger.info("2b. Preprocessing candidate pool (S2+S3 sampled by country for training)...")
    df_cand_clean = _preprocess_candidates_sampled(df_s2, df_s3, CAND_SAMPLE_PER_COUNTRY)
    logger.info(f"   Candidate pool ready: {len(df_cand_clean):,} records")

    # Train/val split
    if val_split > 0:
        logger.info(f"3. Splitting S1 for validation (val_split={val_split})...")
        train_s1, val_s1, train_gt, val_gt = split_validation_data(
            df_s1_clean, df_gt, test_size=val_split
        )
    else:
        train_s1, train_gt = df_s1_clean, df_gt
        val_s1, val_gt = None, None

    # Split cand into S2 and S3 for generate_candidate_pairs interface
    from ..config import ENTITY_ID_COL as EID
    s2_clean = df_cand_clean[df_cand_clean[EID].str.startswith("S2-")]
    s3_clean = df_cand_clean[df_cand_clean[EID].str.startswith("S3-")]

    logger.info("4. Generating candidate pairs for TRAINING split (Blocking stage)...")
    df_pairs_train = generate_candidate_pairs(train_s1, s2_clean, s3_clean)
    logger.info(f"   → {len(df_pairs_train):,} candidate pairs generated.")

    if df_pairs_train.empty:
        logger.error("No candidate pairs generated! Check blocking configuration.")
        return None

    logger.info("5. Building feature matrix for training...")
    df_features_train = build_feature_matrix(df_pairs_train, train_s1, df_cand_clean)

    logger.info("6. Training binary classification model...")
    model = train_matching_model(df_features_train, train_gt)
    logger.info("   → Model saved to models_saved/matching_model.pkl")

    if val_s1 is not None and val_gt is not None:
        logger.info("7. Evaluating on validation split...")
        df_pairs_val = generate_candidate_pairs(val_s1, s2_clean, s3_clean)
        df_features_val = build_feature_matrix(df_pairs_val, val_s1, df_cand_clean)

        val_s1_ids = val_s1[ENTITY_ID_COL]
        df_preds = predict_matches(df_features_val, all_s1_ids=val_s1_ids)

        metrics = evaluate_predictions(df_preds, val_gt)
        logger.info(
            f"   → Validation: "
            f"Precision={metrics['macro_precision']:.4f}  "
            f"Recall={metrics['macro_recall']:.4f}  "
            f"F0.5={metrics['macro_f0.5']:.4f}"
        )
        return model, metrics

    logger.info("Training pipeline finished successfully!")
    return model, {}
