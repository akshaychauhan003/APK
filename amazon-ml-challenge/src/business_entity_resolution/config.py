from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR  = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR  = BASE_DIR / "models_saved"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

OUTPUT_MATCHING   = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

# column names
ID_COL      = "entity_id"
NAME_COL    = "business_name"
ADDR_COL    = "business_address"
COUNTRY_COL = "country"
S1_ID_COL   = "source1_entity_id"
MATCHED_COL = "matched_entity_ids"
CAND_COL    = "candidate_entity_ids"

# model / eval
F_BETA          = 0.5
THRESHOLD       = 0.90
RANDOM_STATE    = 42
BLOCKING_TOP_K  = 15
TRAIN_SAMPLE    = 50_000

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
