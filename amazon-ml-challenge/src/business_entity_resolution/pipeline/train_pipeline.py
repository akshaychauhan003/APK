"""
End-to-end training pipeline orchestrator.

For the large-scale dataset (2.2M+ rows), we train on a random sample
to make the pipeline feasible on a single machine. Blocking + feature
extraction is done on the sample only; the full test set is handled
in the inference pipeline.
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


def run_training_pipeline(val_split: float = 0.20, sample_size: int = TRAIN_SAMPLE_SIZE):
    """Execute end-to-end training and validation pipeline.

    Args:
        val_split:   Fraction of S1 entities held out for validation.
        sample_size: Max S1 entities to use for training (to keep runtime
                     manageable on a single machine). Set to None to use all.
    """
    logger.info("1. Loading raw training data...")
    df_s1, df_s2, df_s3, df_gt = load_all_train_data()
    logger.info(f"   S1={len(df_s1):,}  S2={len(df_s2):,}  S3={len(df_s3):,}  GT={len(df_gt):,}")

    # ── Optional: subsample S1 for fast iteration ──────────────────────────
    if sample_size and len(df_s1) > sample_size:
        logger.info(f"   Sampling {sample_size:,} S1 entities from {len(df_s1):,} for training...")
        df_s1 = df_s1.sample(n=sample_size, random_state=42).reset_index(drop=True)
        sampled_ids = set(df_s1[ENTITY_ID_COL])
        df_gt = df_gt[df_gt[GROUND_TRUTH_S1_COL].isin(sampled_ids)].copy()

    logger.info("2. Preprocessing entity text fields...")
    df_s1_clean = preprocess_dataset(df_s1)
    df_s2_clean = preprocess_dataset(df_s2)
    df_s3_clean = preprocess_dataset(df_s3)

    # Combined candidate pool (S2 + S3)
    df_cand_all = pd.concat([df_s2_clean, df_s3_clean], ignore_index=True)
    df_cand_all = df_cand_all.drop_duplicates(subset=[ENTITY_ID_COL])
    logger.info(f"   Combined candidate pool: {len(df_cand_all):,} records")

    if val_split > 0:
        logger.info(f"3. Splitting dataset for validation (val_split={val_split})...")
        train_s1, val_s1, train_gt, val_gt = split_validation_data(
            df_s1_clean, df_gt, test_size=val_split
        )
    else:
        train_s1, train_gt = df_s1_clean, df_gt
        val_s1, val_gt = None, None

    logger.info("4. Generating candidate pairs for TRAINING split (Blocking stage)...")
    df_pairs_train = generate_candidate_pairs(train_s1, df_s2_clean, df_s3_clean)
    logger.info(f"   → {len(df_pairs_train):,} candidate pairs generated.")

    logger.info("5. Extracting pair similarity features for training...")
    df_features_train = build_feature_matrix(df_pairs_train, train_s1, df_cand_all)

    logger.info("6. Training binary classification model...")
    model = train_matching_model(df_features_train, train_gt)
    logger.info("   → Model saved to models_saved/matching_model.pkl")

    if val_s1 is not None and val_gt is not None:
        logger.info("7. Evaluating model on validation split...")
        df_pairs_val = generate_candidate_pairs(val_s1, df_s2_clean, df_s3_clean)
        df_features_val = build_feature_matrix(df_pairs_val, val_s1, df_cand_all)

        val_s1_ids = val_s1[ENTITY_ID_COL]
        df_preds = predict_matches(df_features_val, all_s1_ids=val_s1_ids)

        metrics = evaluate_predictions(df_preds, val_gt)
        logger.info(
            f"   → Validation Results: "
            f"Precision={metrics['macro_precision']:.4f}  "
            f"Recall={metrics['macro_recall']:.4f}  "
            f"F0.5={metrics['macro_f0.5']:.4f}"
        )

    logger.info("Training pipeline finished successfully!")
    return model
