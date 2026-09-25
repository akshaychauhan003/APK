# Business Entity Resolution

Approach for Amazon ML Challenge 2026 — matching business records across 3 noisy sources (US, India, France).

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src
```

> On Mac, LightGBM needs `brew install libomp`. Without it the pipeline falls back to scikit-learn HistGradientBoostingClassifier automatically.

## Run

```bash
# train + generate output files (recommended: 50k sample)
python src/business_entity_resolution/main.py --mode all --threshold 0.90 --sample-size 50000

# quick test run (5k sample)
python src/business_entity_resolution/main.py --mode all --sample-size 5000

# full dataset training (slow, needs 64GB+ RAM)
python src/business_entity_resolution/main.py --mode train --sample-size 0

# predict only (needs a saved model)
python src/business_entity_resolution/main.py --mode predict --threshold 0.90
```

## Pipeline

1. **Preprocess** — normalise names (lowercase, strip LLC/Pvt/Ltd/Corp etc., `&` → `and`) and addresses (Rd→Road, extract PIN/ZIP codes)
2. **Block** — generate candidate pairs using:
   - exact keys: `(country, first_word_of_name)` and `(country, postal_code)`
   - TF-IDF char n-gram ANN per country (top-15 nearest neighbours)
3. **Features** — 14 similarity scores per pair: Jaccard, fuzzy ratio, token sort/set ratio, Soundex/Metaphone phonetic, digit-set Jaccard, postal and country exact match
4. **Model** — LightGBM binary classifier (XGBoost / HistGBM fallback) with class-weight balancing for heavy imbalance
5. **Output** — `output/matching_results.tsv` and `output/candidate_pairs.tsv`

## Arguments

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `all` | `train` / `predict` / `all` |
| `--threshold` | `0.90` | Higher = more precise (F_0.5 favours precision) |
| `--sample-size` | `50000` | S1 entities to train on; `0` = full dataset |
| `--val-split` | `0.20` | Fraction held out for validation |

## Validate before submitting

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Package submission zip

```bash
python package.py --team-name APK
```

Produces `APK_submission.zip` with the required structure.

## Notes

- Dataset lives in `dataset/` (symlinked from `../student_resource/dataset/`)
- Trained model saves to `models_saved/matching_model.pkl`
- France appears only in the test set — the pipeline handles it automatically (no country hardcoding)
