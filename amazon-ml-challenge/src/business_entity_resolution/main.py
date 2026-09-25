"""
Main entry point for Business Entity Resolution pipeline.

Usage examples:
  # Full pipeline (train on sample + predict on all test):
  python3 src/business_entity_resolution/main.py --mode all --threshold 0.90

  # Train only (with validation):
  python3 src/business_entity_resolution/main.py --mode train --val-split 0.20

  # Predict only (requires a saved model):
  python3 src/business_entity_resolution/main.py --mode predict --threshold 0.90

  # Train on full dataset (slow — only do this once):
  python3 src/business_entity_resolution/main.py --mode train --sample-size 0
"""

import argparse
import sys
import logging

from business_entity_resolution.pipeline.train_pipeline import run_training_pipeline
from business_entity_resolution.pipeline.inference_pipeline import run_inference_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge: Business Entity Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "all"],
        default="all",
        help="Pipeline execution mode: train, predict, or all (default: all)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20,
        help="Validation split ratio for training (default: 0.20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Classification decision threshold (default: 0.90 — precision-heavy for F_0.5)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50_000,
        help=(
            "Number of S1 training entities to sample for fast iteration. "
            "Set to 0 to use the full dataset (default: 50000)"
        ),
    )

    args = parser.parse_args()

    sample_size = args.sample_size if args.sample_size > 0 else None

    if args.mode in ["train", "all"]:
        logger.info("Executing training pipeline...")
        if sample_size:
            logger.info(f"  Training on a SAMPLE of {sample_size:,} S1 entities.")
        else:
            logger.info("  Training on the FULL dataset (this may take a long time).")
        run_training_pipeline(val_split=args.val_split, sample_size=sample_size)

    if args.mode in ["predict", "all"]:
        logger.info("Executing inference pipeline...")
        run_inference_pipeline(threshold=args.threshold)


if __name__ == "__main__":
    main()
