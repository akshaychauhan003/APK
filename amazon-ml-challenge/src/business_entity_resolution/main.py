import argparse
import logging
import sys

from business_entity_resolution.pipeline.train_pipeline import run_training
from business_entity_resolution.pipeline.inference_pipeline import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    p.add_argument("--mode",        choices=["train", "predict", "all"], default="all")
    p.add_argument("--threshold",   type=float, default=0.90,
                   help="match decision threshold (default 0.90)")
    p.add_argument("--sample-size", type=int,   default=50_000,
                   help="S1 entities to sample for training, 0=full (default 50000)")
    p.add_argument("--val-split",   type=float, default=0.20,
                   help="validation fraction (default 0.20)")
    args = p.parse_args()

    sample = args.sample_size if args.sample_size > 0 else None

    if args.mode in ("train", "all"):
        log.info(f"=== training (sample={sample or 'full'}) ===")
        _, metrics = run_training(val_split=args.val_split, sample_size=sample)
        if metrics:
            log.info(
                f"validation → "
                f"P={metrics.get('macro_precision', 0):.4f}  "
                f"R={metrics.get('macro_recall', 0):.4f}  "
                f"F0.5={metrics.get('macro_f0.5', 0):.4f}"
            )

    if args.mode in ("predict", "all"):
        log.info("=== inference ===")
        run_inference(threshold=args.threshold)


if __name__ == "__main__":
    sys.exit(main())
