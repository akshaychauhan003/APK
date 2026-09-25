"""
Configuration settings for Business Entity Resolution pipeline.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models_saved"

# Dataset Files
TRAIN_S1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1_PATH = TEST_DIR / "test_source1.tsv"
TEST_S2_PATH = TEST_DIR / "test_source2.tsv"
TEST_S3_PATH = TEST_DIR / "test_source3.tsv"

OUTPUT_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Column Schema
ENTITY_ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

GROUND_TRUTH_S1_COL = "source1_entity_id"
GROUND_TRUTH_MATCHED_COL = "matched_entity_ids"

MATCHING_HEADER = [GROUND_TRUTH_S1_COL, GROUND_TRUTH_MATCHED_COL]
CANDIDATE_HEADER = [GROUND_TRUTH_S1_COL, "candidate_entity_ids"]

# Blocking Parameters
BLOCKING_TOP_K = 15
MIN_BLOCKING_SIMILARITY = 0.25

# Model & Evaluation Parameters
F_BETA = 0.5
# IMPORTANT: F_0.5 weights precision 2x over recall.
# Use a HIGH threshold (0.85-0.95) so the model only says "Match"
# when it is very confident. False merges are catastrophic.
CLASSIFICATION_THRESHOLD = 0.90
RANDOM_STATE = 42

# Ensure required directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
