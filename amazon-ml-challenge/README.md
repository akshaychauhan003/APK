# Amazon ML Challenge 2026: Business Entity Resolution

End-to-end ML pipeline for resolving multi-source business entities across noisy, fragmented commercial datasets (2.2M+ train, 1.7M+ test entities, US + India + France).

---

## 📁 Repository Structure

```
amazon-ml-challenge/
│
├── dataset/
│   ├── train/                         → symlinks to student_resource/dataset/train/
│   │   ├── train_source1.tsv          (2.2M entities — S1)
│   │   ├── train_source2.tsv          (5.0M entities — S2)
│   │   ├── train_source3.tsv          (5.3M entities — S3)
│   │   └── train_ground_truth.tsv     (2.2M rows — source1_entity_id → matched_entity_ids)
│   └── test/                          → symlinks to student_resource/dataset/test/
│       ├── test_source1.tsv           (1.7M entities)
│       ├── test_source2.tsv           (4.9M entities)
│       └── test_source3.tsv           (5.1M entities)
│
├── src/
│   └── business_entity_resolution/
│       ├── config.py                  ← Central config (paths, thresholds, sample sizes)
│       ├── data/                      ← TSV loaders (always sep='\t')
│       ├── preprocessing/             ← Name normalization, address expansion
│       ├── blocking/                  ← Exact + TF-IDF blocking (candidate pair generation)
│       ├── features/                  ← 14-dimensional similarity feature vector (vectorized)
│       ├── models/                    ← Auto-selects LightGBM → XGBoost → HistGBM → GBM
│       ├── evaluation/                ← Macro F_0.5 scorer
│       ├── pipeline/                  ← Train + Inference orchestrators (chunked for scale)
│       └── main.py                    ← Single entry point
│
├── output/
│   ├── matching_results.tsv           ← Final leaderboard file (1 row per S1 entity)
│   └── candidate_pairs.tsv            ← Blocking stage output (1 row per S1 entity)
│
├── utils/
│   └── validate_submission.py         ← Official validator (run before every submission!)
│
├── notebooks/
│   └── eda.py                         ← Exploratory Data Analysis script
│
├── package_submission.py              ← Builds the submission zip (run after pipeline)
├── README.md
├── requirements.txt
├── PROJECT_PLAN.md
└── Documentation_template.md
```

---

## ⚙️ Installation & Setup

> **Note:** The actual dataset files live in `../student_resource/dataset/` and are accessed via symlinks in `dataset/`. The symlinks are already set up.

**1. Create and activate a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

> If LightGBM fails to load (`libomp.dylib` missing on macOS), install it with:
> ```bash
> brew install libomp
> ```
> Otherwise the pipeline automatically falls back to **scikit-learn HistGradientBoostingClassifier** — no action needed.

**3. Set PYTHONPATH:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

---

## 🚀 Execution

### Quick Start (Fast — 5k sample for testing)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/business_entity_resolution/main.py --mode all --threshold 0.90 --sample-size 5000
```

### Full Training Run (50k sample — recommended for submission)
```bash
python3 src/business_entity_resolution/main.py --mode all --threshold 0.90 --sample-size 50000
```

### Train on Complete Dataset (2.2M — very slow, requires 64GB+ RAM)
```bash
python3 src/business_entity_resolution/main.py --mode train --sample-size 0
```

### Run EDA on Training Data
```bash
python3 notebooks/eda.py
```

---

## 🧪 Validate Before Submitting (MANDATORY)

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Exit code 0 = safe to submit. Any errors **must** be fixed first.

---

## 📦 Create Submission Zip

After the pipeline runs and validation passes:

```bash
python3 package_submission.py --team-name "YourTeamName"
```

This produces `YourTeamName_submission.zip` with the required structure:
```
YourTeamName_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── utils/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 🧩 Module Overview

| Subsystem | Module | Description |
|---|---|---|
| **Data Loading** | `data/loader.py` | Reads TSVs with `sep='\t'`, `dtype=str`, `keep_default_na=False` |
| **Preprocessing** | `preprocessing/` | Name normalization (legal suffixes, Unicode), address expansion (abbreviations, postal extraction) |
| **Blocking** | `blocking/` | Exact key blocks (country+first_word, country+postal) + TF-IDF char n-gram ANN (top-15 per entity) |
| **Feature Extraction** | `features/` | 14 vectorized features: Jaccard, RapidFuzz edit, phonetic (Soundex/Metaphone), postal/country match |
| **Model** | `models/` | Auto-selects best backend: LightGBM → XGBoost → HistGBM → GBM |
| **Evaluation** | `evaluation/` | Macro F_0.5, Precision, Recall on validation split |
| **Pipeline** | `pipeline/` | Chunked inference (50k S1 at a time) to handle 1.7M test set without OOM |

---

## 🎯 Scoring Strategy

- **Metric:** Macro F_0.5 (precision weighted **2×** more than recall)
- **Default threshold:** `0.90` — only call "Match" when 90%+ confident
- **Singletons:** Empty `matched_entity_ids` always emits score = 1.0 ✅
- **Never hard-code country logic** — test set includes France (unseen in training)
- **No external APIs** — disqualification is instant if used

---

## ⚡ Command Cheat Sheet

```bash
# Set PYTHONPATH (always needed)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Fast test run (5k sample)
python3 src/business_entity_resolution/main.py --mode all --sample-size 5000

# Full submission run (50k sample)
python3 src/business_entity_resolution/main.py --mode all --sample-size 50000 --threshold 0.90

# Validate output
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

# Package submission
python3 package_submission.py --team-name "YourTeamName"
```
