# Business Entity Resolution: Project Plan & Status

> **Last Updated:** 2026-09-25  
> **Dataset:** 2.2M train S1 · 5M S2 · 5.3M S3 · 1.7M test S1 — US + India + France

---

## 👥 Team Roles

| Role | Owner | Key Files | Output |
|---|---|---|---|
| **Role 1 — Data & Blocking** | — | `src/blocking/`, `src/preprocessing/` | `output/candidate_pairs.tsv` |
| **Role 2 — Feature & Model** | — | `src/features/`, `src/models/`, `src/pipeline/` | `output/matching_results.tsv` |
| **Role 3 — Validation & Docs** | — | `utils/validate_submission.py`, `Documentation_template.md` | Clean final zip |

---

## 📅 Phases & Completion Status

### ✅ Phase 1: Foundation (COMPLETE)
- [x] Located real dataset at `../student_resource/dataset/` — symlinked into `dataset/train/` and `dataset/test/`
- [x] Verified `sep='\t'` on all TSV reads (`keep_default_na=False`, `dtype=str`)
- [x] Confirmed schema: columns `entity_id`, `business_name`, `business_address`, `country`
- [x] Confirmed country values: `"US"`, `"India"` in train; `"France"` appears in test only
- [x] EDA script ready: `notebooks/eda.py` — run before every modeling iteration
- [x] Git repo set up with `.gitignore` for `*.pkl`, `__pycache__`, `*.pyc`, `models_saved/`

**Key findings from real dataset inspection:**
- S1 has **2,206,821** entities; GT covers all S1 rows (singletons have blank `matched_entity_ids`)
- **Singletons**: 123,247 out of 2,206,821 (~5.6%) — each scores 1.0 automatically
- **Entities with matches**: 2,083,574 (~94.4%) — most S1 entities have at least one match
- **Max matched IDs per entity**: 11 (can match to multiple S2 + S3 records simultaneously)
- **Test country distribution**: 809,986 India · 663,106 US · 259,452 France
- **France is 15% of test set** but absent from training — TF-IDF blocking must handle it
- Ground truth example: `S1-965667 → S2-681193310,S2-743505751,S3-775321672,...`
- Data has Unicode (Hindi script `राम मार्केटिंग`, French accents) — handled by NFKD normalization
- Many addresses missing postal codes — postal blocking is supplementary, not primary
- Common noise: `-- Holloway Peak Inc Seafood` (prefix garbage), empty addresses

---

### ✅ Phase 2: Blocking & Candidates (COMPLETE)

**Blocking Strategy (3-layer hybrid):**

| Layer | Method | Key | Handles |
|---|---|---|---|
| Exact Name Block | `(country, first_word_of_clean_name)` | `exact_blocking.py` | Abbreviations, suffix variants |
| Exact Postal Block | `(country, postal_code)` | `exact_blocking.py` | Location-confirmed matches |
| TF-IDF Fuzzy Block | Char n-gram ANN (top-15 per entity) | `fuzzy_blocking.py` | Typos, reorderings, novel countries |

- [x] `src/blocking/exact_blocking.py` — vectorized with `itertuples()`, country+name+postal blocking
- [x] `src/blocking/fuzzy_blocking.py` — per-country TF-IDF char n-gram cosine ANN
- [x] `src/blocking/candidate_generation.py` — combines both, formats `candidate_pairs.tsv`
- [x] `output/candidate_pairs.tsv` format confirmed: `source1_entity_id \t candidate_entity_ids` (comma-separated)

---

### ✅ Phase 3: Feature Engineering & Modeling (COMPLETE)

**14-Dimensional Feature Vector:**

| Feature | Type | Description |
|---|---|---|
| `name_jaccard` | Name | Token set Jaccard over normalized names |
| `name_first_word_match` | Name | Binary: first tokens identical |
| `name_fuzz_ratio` | Name | RapidFuzz character edit distance |
| `name_token_sort_ratio` | Name | Word-order invariant edit distance |
| `name_token_set_ratio` | Name | Subset-tolerant edit distance |
| `name_len_diff` | Name | Normalized length difference |
| `name_soundex_match` | Phonetic | Soundex match (catches Corp/Korp typos) |
| `name_metaphone_match` | Phonetic | Metaphone match (double-consonant variants) |
| `addr_jaccard` | Address | Token Jaccard over cleaned addresses |
| `postal_match` | Address | Exact=1.0, one missing=0.5, mismatch=0.0 |
| `country_match` | Address | Binary: same country string |
| `digit_jaccard` | Address | Jaccard over numeric tokens in address |
| `addr_fuzz_ratio` | Address | Full address edit distance |
| `addr_token_set_ratio` | Address | Address token set ratio |

**Model: Auto-selecting gradient boosting backend**
```
Priority: LightGBM → XGBoost → HistGradientBoostingClassifier → GradientBoostingClassifier
```
- `libomp` missing on macOS without Homebrew → auto-falls back to sklearn `HistGBM` (no native deps)
- Install `brew install libomp` for LightGBM/XGBoost performance
- Threshold: **0.90** (precision-heavy — F_0.5 penalizes false positives 2×)

- [x] `src/features/feature_builder.py` — **fully vectorized** (no `iterrows()`), bulk numpy ops
- [x] `src/models/model.py` — auto-backend selection with graceful fallback
- [x] `src/models/train.py` — label creation from ground truth, model fit + save
- [x] `src/models/predict.py` — threshold-based filtering, singleton handling
- [x] `src/evaluation/metrics.py` — Macro F_0.5, Precision, Recall
- [x] `src/evaluation/validation.py` — stratified train/val split

**Scale considerations (CRITICAL for 2.2M dataset):**
- Training uses a **sample** (default 50k S1 entities) — configurable via `--sample-size`
- Inference processes test S1 in **chunks of 50k** to avoid OOM on 1.7M entities
- S2+S3 preprocessing is done **once** then reused across all chunks

---

### 🔄 Phase 4: Final Packaging (IN PROGRESS)

**Critical fixes applied after server restart:**
- [x] `exact_blocking.py` — **rewritten with pandas merge** (was itertuples loop, 1000× faster on 10M rows)
- [x] `train_pipeline.py` — **S2+S3 sub-sampled per country** (300k cap) before preprocessing, eliminates 9-min bottleneck
- [x] `candidate_generation.py` — 2M pair safety cap + better logging
- [x] `main.py` — handles `(model, metrics)` tuple return from training
- [x] Pipeline restarted (task-253) with corrected code

**Remaining to complete:**
- [ ] **Await pipeline completion** → review `output/matching_results.tsv`
- [ ] Run `python3 utils/validate_submission.py` — verify PASS
- [ ] Record actual F_0.5 validation score in `Documentation_template.md`
- [ ] Rerun with `--sample-size 50000` for submission-quality model
- [ ] Run `python3 package_submission.py --team-name "YourTeamName"` → create final zip
- [ ] Commit all changes to git

---

## 🚨 Golden Rules (Never Break These)

1. **NO external APIs.** No geocoding. No web lookups. Instant disqualification.
2. **Always `sep='\t'`** when reading or writing any `.tsv` file.
3. **Every S1 entity must have exactly one row** in `matching_results.tsv` (even singletons).
4. **Singletons** get an **empty** `matched_entity_ids` — they score 1.0 each. Never omit them.
5. **Never hard-code country logic** — France appears in test but not in training.
6. **Run the validator** (`utils/validate_submission.py`) before every submission.

---

## 🗂️ Final File Structure

```
amazon-ml-challenge/
│
├── dataset/
│   ├── train/                         → symlinks to ../student_resource/dataset/train/
│   └── test/                          → symlinks to ../student_resource/dataset/test/
│
├── src/business_entity_resolution/
│   ├── config.py                      TRAIN_SAMPLE_SIZE=50k, THRESHOLD=0.90, BLOCKING_TOP_K=15
│   ├── data/                          loader.py (sep='\t'), validator.py
│   ├── preprocessing/                 preprocess.py (vectorized), normalize_names, normalize_addresses
│   ├── blocking/                      exact_blocking.py, fuzzy_blocking.py, candidate_generation.py
│   ├── features/                      feature_builder.py (vectorized, 14 features)
│   ├── models/                        model.py (auto-backend), train.py, predict.py
│   ├── evaluation/                    metrics.py (F0.5), validation.py (split)
│   ├── pipeline/                      train_pipeline.py (sampled), inference_pipeline.py (chunked)
│   └── main.py                        --mode [train|predict|all] --sample-size --threshold
│
├── output/
│   ├── matching_results.tsv           ← 1 row per S1 entity (scored on leaderboard)
│   └── candidate_pairs.tsv            ← 1 row per S1 entity (blocking output)
│
├── utils/
│   └── validate_submission.py         ← Official validator (MUST run before every submission)
│
├── notebooks/
│   └── eda.py                         ← EDA script (run first for noise analysis)
│
├── package_submission.py              ← Builds final submission zip
├── README.md                          ← Full setup and run instructions
├── requirements.txt                   ← pandas, sklearn, rapidfuzz, jellyfish, lightgbm, xgboost
├── PROJECT_PLAN.md                    ← This file
└── Documentation_template.md          ← Methodology writeup (filled)
```

---

## 🧩 Module Dependency Map

```
main.py  (--mode all --sample-size 50000 --threshold 0.90)
 │
 ├── pipeline/train_pipeline.py
 │     ├── data/loader.py              reads TSVs with sep='\t'
 │     ├── preprocessing/preprocess.py VECTORIZED — name normalization, address expansion
 │     ├── blocking/                   exact (name+postal) + TF-IDF fuzzy
 │     ├── features/feature_builder.py VECTORIZED — 14-dim similarity vector
 │     └── models/train.py             LightGBM / XGBoost / HistGBM (auto-select)
 │
 └── pipeline/inference_pipeline.py   CHUNKED — 50k S1 at a time
       ├── preprocessing/              S2+S3 preprocessed once, reused per chunk
       ├── blocking/                   per-chunk candidate generation
       ├── features/                   per-chunk feature matrix
       ├── models/predict.py           threshold=0.90, singletons → empty string
       └── output/                     matching_results.tsv + candidate_pairs.tsv
```

---

## 🎯 Scoring Notes

| Concept | Detail |
|---|---|
| **Metric** | Macro F_0.5 — precision weighted **2×** over recall |
| **Threshold** | `0.90` — only call "Match" when 90%+ confident |
| **False positives** | Catastrophic — avoid at all costs |
| **Singletons** | Always emit, always score 1.0 |
| **France** | Test-only country — TF-IDF blocking handles it without hard-coding |
| **Max sample size** | 50k for training → raises to full 2.2M if time/RAM allows |

---

## ⚡ Quick Run Commands

```bash
# 0. Setup (once)
cd amazon-ml-challenge/
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pip install -r requirements.txt

# 1. EDA (understand the data first)
python3 notebooks/eda.py

# 2. Fast test run (5k sample, ~5 min)
python3 src/business_entity_resolution/main.py --mode all --sample-size 5000 --threshold 0.90

# 3. Submission-quality run (50k sample, ~30-60 min)
python3 src/business_entity_resolution/main.py --mode all --sample-size 50000 --threshold 0.90

# 4. Validate output (MANDATORY before every submission)
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 5. Package submission zip
python3 package_submission.py --team-name "YourTeamName"
```
