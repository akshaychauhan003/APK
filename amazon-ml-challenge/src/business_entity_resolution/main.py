"""
Main entry point for Business Entity Resolution pipeline.
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
    parser = argparse.ArgumentParser(description="Amazon ML Challenge: Business Entity Resolution")
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
        default=0.50,
        help="Classification decision threshold (default: 0.50)",
    )

    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        logger.info("Executing training pipeline...")
        run_training_pipeline(val_split=args.val_split)

    if args.mode in ["predict", "all"]:
        logger.info("Executing inference pipeline...")
        run_inference_pipeline(threshold=args.threshold)


if __name__ == "__main__":
    main()
