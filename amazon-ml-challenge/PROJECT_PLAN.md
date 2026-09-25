# Business Entity Resolution: Project Plan & Phases

## 👥 Team Roles
- **Role 1 (Data & Blocking):** Owns `src/business_entity_resolution/blocking/`. Outputs `output/candidate_pairs.tsv`.
- **Role 2 (Feature & Model):** Owns `src/business_entity_resolution/models/`. Outputs `output/matching_results.tsv`.
- **Role 3 (Validation & Docs):** Owns `utils/validate_submission.py` and `Documentation_template.md`.

## 📅 Phases

### Phase 1: Foundation (Days 1-2)
- [ ] Download and unzip datasets into `dataset/` folder.
- [ ] Verify `sep='\t'` on all `.tsv` files.
- [ ] Perform EDA. Map out noise patterns (abbreviations, missing components).
- [ ] Set up Git repo. Create `dev` branch. No one pushes to `main` directly.

### Phase 2: Blocking & Candidates (Days 3-5)
- [ ] Role 1: Build blocking strategy (TF-IDF, MinHash, or rule-based).
- [ ] Role 1: Output `candidate_pairs.tsv` into `output/`.
- [ ] Role 2: Build feature engineering pipeline using the candidates.
- [ ] Role 3: Build F_0.5 evaluation script on a validation split.

### Phase 3: Modeling & Tuning (Days 6-8)
- [ ] Role 2: Train precision-heavy model (XGBoost/Transformers).
- [ ] Role 2: Tune threshold to maximize F_0.5 (penalize False Positives heavily).
- [ ] Role 1: Optimize blocking to increase recall ceiling without exploding candidates.
- [ ] Role 3: Run `validate_submission.py` on model outputs daily.

### Phase 4: Final Packaging (Days 9-10)
- [ ] Role 2: Generate final `matching_results.tsv`.
- [ ] Role 3: Fill out `Documentation_template.md`.
- [ ] Role 3: Ensure `code/` folder is self-contained and runs end-to-end.
- [ ] ALL: Freeze code. Zip the submission. Run validator one last time.
- [ ] ALL: Submit the zip and the leaderboard TSV.

## 🚨 Golden Rules (Never Break These)
1. NO external data lookup. NO APIs. Disqualification is instant.
2. Always use `sep='\t'` when reading or writing files.
3. Every Source 1 entity must have exactly one row in the final output.
4. Leave `matched_entity_ids` empty for singletons.

---

## 🗂️ File Structure Reference

```
amazon-ml-challenge/
│
├── dataset/
│   ├── train/                        # Raw training TSVs (sep='\t')
│   └── test/                         # Raw test TSVs (sep='\t')
│
├── src/business_entity_resolution/
│   ├── config.py                     # Central config — paths, thresholds
│   ├── data/                         # loader.py, validator.py
│   ├── preprocessing/                # normalize_names, normalize_addresses
│   ├── blocking/                     # exact_blocking, fuzzy_blocking, candidate_generation
│   ├── features/                     # name_features, address_features, feature_builder
│   ├── models/                       # model.py, train.py, predict.py
│   ├── evaluation/                   # metrics.py (F0.5), validation.py (split)
│   ├── pipeline/                     # train_pipeline.py, inference_pipeline.py
│   └── main.py                       # Single entry point (--mode train/predict/all)
│
├── output/
│   ├── matching_results.tsv          # Leaderboard file — 1 row per S1 entity
│   └── candidate_pairs.tsv           # Blocking output — 1 row per S1 entity
│
├── utils/
│   └── validate_submission.py        # MUST run before every submission
│
├── PROJECT_PLAN.md                   # This file
├── README.md                         # How to reproduce results
├── requirements.txt                  # Pinned dependencies
└── Documentation_template.md        # Fill before final submission
```

## 🧩 Module Dependency Map

```
main.py
 ├── pipeline/train_pipeline.py
 │     ├── data/loader.py          → reads TSVs with sep='\t'
 │     ├── preprocessing/          → normalize names & addresses
 │     ├── blocking/               → candidate pair generation
 │     ├── features/               → similarity feature matrix
 │     └── models/train.py         → LightGBM binary classifier
 │
 └── pipeline/inference_pipeline.py
       ├── data/loader.py
       ├── preprocessing/
       ├── blocking/
       ├── features/
       ├── models/predict.py       → threshold = 0.90 (precision-heavy)
       └── output/                 → matching_results.tsv + candidate_pairs.tsv
```

## 🎯 Scoring Notes
- **Metric:** Macro F_0.5 (precision weighted 2× more than recall)
- **Strategy:** Tune threshold to ~0.90 — only call "Match" when 90%+ confident
- **Singletons:** Always emit an empty `matched_entity_ids` for no-match entities — these score 1.0 each!
- **France:** Test set includes France (unseen in training). Never hard-code country logic.

## 🔧 Quick Run Commands

```bash
# Install deps
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Full pipeline (train + predict)
python3 src/business_entity_resolution/main.py --mode all --threshold 0.90

# Train only
python3 src/business_entity_resolution/main.py --mode train --val-split 0.20

# Predict only (requires trained model)
python3 src/business_entity_resolution/main.py --mode predict --threshold 0.90

# Validate before submitting
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
