# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains an end-to-end Machine Learning pipeline for resolving multi-source business entities in noisy, fragmented commercial datasets.

---

## 📁 Repository Structure

```
amazon-ml-challenge/
│
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   └── business_entity_resolution/
│       │
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── validator.py
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── normalize_names.py
│       │   ├── normalize_addresses.py
│       │   └── preprocess.py
│       │
│       ├── blocking/
│       │   ├── __init__.py
│       │   ├── exact_blocking.py
│       │   ├── fuzzy_blocking.py
│       │   └── candidate_generation.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── name_features.py
│       │   ├── address_features.py
│       │   └── feature_builder.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── model.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── validation.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── train_pipeline.py
│       │   └── inference_pipeline.py
│       │
│       └── main.py
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── utils/
│   └── validate_submission.py
│
├── README.md
├── requirements.txt
└── Documentation_template.md
```

---

## ⚙️ Installation & Setup

1. **Environment Setup:**
   Ensure Python 3.8+ is installed. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set PYTHONPATH:**
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   ```

---

## 🚀 Execution & Usage

### 1. Run Complete End-to-End Pipeline (Train + Predict)
```bash
python3 src/business_entity_resolution/main.py --mode all
```

### 2. Train Model Only
```bash
python3 src/business_entity_resolution/main.py --mode train --val-split 0.20
```

### 3. Generate Submission Predictions Only
```bash
python3 src/business_entity_resolution/main.py --mode predict --threshold 0.50
```

---

## 🧪 Submission Validation

Before submitting your results, validate the output files using the official submission validator:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 🧩 Module Overview

| Subsystem | File / Module | Function & Purpose |
| :--- | :--- | :--- |
| **Data Loading & Validation** | `data/loader.py`, `data/validator.py` | Load TSV datasets (`sep="\t"`) and validate entity schema (`S1-`, `S2-`, `S3-` prefixes) |
| **Text Preprocessing** | `preprocessing/` | Clean business names (legal suffix removal) and normalize addresses (abbreviation expansion) |
| **Blocking Stage** | `blocking/` | Hybrid candidate pair generation (Exact key matching + TF-IDF character n-gram cosine similarity) |
| **Feature Extraction** | `features/` | Calculate pair similarity vector (Jaccard, RapidFuzz edit distance, postal code & country match) |
| **Matching Model** | `models/` | Train LightGBM binary classifier and compute match probabilities |
| **Evaluation** | `evaluation/` | Compute Macro $F_{0.5}$ score, Precision, and Recall |
| **Pipeline** | `pipeline/` | Orchestrate training and inference execution |
