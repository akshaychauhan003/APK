"""
End-to-end training pipeline orchestrator.
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
from ..config import ENTITY_ID_COL, GROUND_TRUTH_S1_COL

logger = logging.getLogger(__name__)


def run_training_pipeline(val_split: float = 0.20):
    """Execute end-to-end training and validation pipeline."""
    logger.info("1. Loading raw training data...")
    df_s1, df_s2, df_s3, df_gt = load_all_train_data()

    logger.info("2. Preprocessing entity text fields...")
    df_s1_clean = preprocess_dataset(df_s1)
    df_s2_clean = preprocess_dataset(df_s2)
    df_s3_clean = preprocess_dataset(df_s3)

    # Combine S2+S3 candidates — do NOT use deprecated _append()
    df_cand_all = pd.concat([df_s2_clean, df_s3_clean], ignore_index=True)
    df_cand_all = df_cand_all.drop_duplicates(subset=[ENTITY_ID_COL])

    if val_split > 0:
        logger.info(f"3. Splitting dataset for validation (val_split={val_split})...")
        # split_validation_data expects entity_id column in df_s1_clean
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

        # Align predictions to ground truth column naming
        metrics = evaluate_predictions(df_preds, val_gt)
        logger.info(
            f"   → Validation Results: "
            f"Precision={metrics['macro_precision']:.4f}  "
            f"Recall={metrics['macro_recall']:.4f}  "
            f"F0.5={metrics['macro_f0.5']:.4f}"
        )

    logger.info("Training pipeline finished successfully!")
    return model

