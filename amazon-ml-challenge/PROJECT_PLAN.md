# Business Entity Resolution: Project Plan & Status

> **Last Updated:** 2026-09-25 14:38 IST  
> **Dataset:** 2.2M train S1 · 5M S2 · 5.3M S3 · 1.7M test S1 — US + India + France  
> **Best F_0.5:** **0.8642** (P=0.8776, R=0.8653) — 5k sample, threshold=0.90, HistGBM

---

## 👥 Team Roles

| Role | Key Files | Output |
|---|---|---|
| **Role 1 — Data & Blocking** | `src/blocking/`, `src/preprocessing/` | `output/candidate_pairs.tsv` |
| **Role 2 — Feature & Model** | `src/features/`, `src/models/`, `src/pipeline/` | `output/matching_results.tsv` |
| **Role 3 — Validation & Docs** | `utils/validate_submission.py`, `Documentation_template.md` | Validated zip |

---

## 📅 Phases & Completion Status

### ✅ Phase 1: Foundation (COMPLETE)
- [x] Real dataset at `../student_resource/dataset/` — symlinked into `dataset/train/` and `dataset/test/`
- [x] `sep='\t'` on all TSV reads (`keep_default_na=False`, `dtype=str`)
- [x] Schema: `entity_id`, `business_name`, `business_address`, `country`
- [x] Country values: `"US"`, `"India"` in train; `"France"` in test only (15% of test!)
- [x] EDA script: `notebooks/eda.py`
- [x] Git repo with `.gitignore` — models, pycache, zips excluded

**Real EDA Findings:**
| Stat | Value |
|---|---|
| Train S1 entities | 2,206,821 |
| Train S2 entities | 5,034,616 |
| Train S3 entities | 5,285,603 |
| Test S1 entities | 1,732,544 |
| Test S2 entities | 4,887,273 |
| Test S3 entities | 5,082,316 |
| Singletons in train GT | 123,247 (~5.6%) — score 1.0 each |
| Entities with matches | 2,083,574 (~94.4%) |
| Max matches per entity | 11 |
| Test: India / US / France | 809,986 / 663,106 / 259,452 |

---

### ✅ Phase 2: Blocking & Candidates (COMPLETE)

**3-Layer Hybrid Blocking:**

| Layer | Key | Note |
|---|---|---|
| Exact Name Block | `(country_lc, first_word)` — stopwords filtered | Cuts generic-word explosion |
| Exact Postal Block | `(country_lc, postal_code)` | Only when postal is non-empty |
| TF-IDF Fuzzy Block | Char n-gram (2-4), cosine ANN, top-15 per entity | 200k candidate cap per country, 50k vocab |

**Performance (5k S1 sample):**
- Exact name block: ~1M pairs (merge in **4 seconds** — pandas merge, not loops)
- TF-IDF India: ~24k fuzzy pairs | TF-IDF US: ~60k fuzzy pairs
- Total pairs: ~1.3M → reduced to ~1M after stopword filter active next run

**Files:**
- [x] `exact_blocking.py` — **pandas merge** (1000× faster than itertuples on 10M rows), stopword filter
- [x] `fuzzy_blocking.py` — memory-safe (200k cap/country, 50k vocab, batched kNN, n_jobs=-1)
- [x] `candidate_generation.py` — 2M pair cap, dedup, format for TSV output

---

### ✅ Phase 3: Feature Engineering & Modeling (COMPLETE)

**14-Dimensional Feature Vector (fully vectorized — no iterrows):**

| Feature | Type | Description |
|---|---|---|
| `name_jaccard` | Name | Token set Jaccard over normalized names |
| `name_first_word_match` | Name | Binary: first tokens identical |
| `name_fuzz_ratio` | Name | RapidFuzz character edit distance |
| `name_token_sort_ratio` | Name | Word-order invariant edit distance |
| `name_token_set_ratio` | Name | Subset-tolerant edit distance |
| `name_len_diff` | Name | Normalized length difference |
| `name_soundex_match` | Phonetic | Soundex (catches Corp/Korp typos) |
| `name_metaphone_match` | Phonetic | Metaphone (double-consonant variants) |
| `addr_jaccard` | Address | Token Jaccard over cleaned addresses |
| `postal_match` | Address | Exact=1.0, one missing=0.5, mismatch=0.0 |
| `country_match` | Address | Binary: same country string |
| `digit_jaccard` | Address | Jaccard over numeric tokens in address |
| `addr_fuzz_ratio` | Address | Full address edit distance |
| `addr_token_set_ratio` | Address | Address token set ratio |

**Model Backend (auto-select):**
```
LightGBM → XGBoost → HistGradientBoostingClassifier → GradientBoostingClassifier
```
- macOS without Homebrew → HistGBM (no libomp needed) — **currently active**
- `brew install libomp` → unlocks LightGBM for 3-5× speedup
- Threshold: **0.90** — precision-heavy for F_0.5 metric

**Training key fix — Positive-Aware Sampling:**
> Critical bug found and fixed: random S2/S3 sampling gave only 728 positives / 1.3M pairs
> (1808:1 imbalance — model learned nothing). Fix: always include all true-match S2/S3
> entity IDs from GT in the candidate pool, then fill remaining slots with random negatives.
> Result: 12,504 positives / 1,069,423 negatives (85:1) → F_0.5 jumped from 0.12 to **0.86**.

**Files:**
- [x] `feature_builder.py` — vectorized numpy/pandas bulk ops, 14 features
- [x] `model.py` — auto-backend, translates `scale_pos_weight` to `class_weight` for sklearn
- [x] `train.py` — vectorized label creation, class imbalance detection + weighting
- [x] `predict.py` — threshold=0.90, singletons → empty string
- [x] `metrics.py` — Macro F_0.5, Precision, Recall
- [x] `validation.py` — stratified train/val split

---

### 🔄 Phase 4: Final Packaging (IN PROGRESS)

**All Code Fixes Applied & Pushed to GitHub:**
- [x] `exact_blocking.py` — pandas merge (1000× faster), stopword filter on name keys
- [x] `fuzzy_blocking.py` — memory-safe (200k/country cap, batched kNN)
- [x] `candidate_generation.py` — 2M pair cap, dedup, clean logging
- [x] `train_pipeline.py` — positive-aware candidate sampling (17,383 guaranteed true matches)
- [x] `preprocess.py` — fully vectorized (str.replace, str.extract — no apply/iterrows)
- [x] `model.py` — auto-backend + proper class_weight translation
- [x] `train.py` — vectorized labels + imbalance detection
- [x] `main.py` — (model, metrics) tuple handling

**Training Results (5k sample, validated ✅):**
| Metric | Value |
|---|---|
| F_0.5 (macro) | **0.8642** |
| Precision | 0.8776 |
| Recall | 0.8653 |
| Positives in train | 12,504 |
| Imbalance ratio | 85:1 |
| Backend | HistGBM (sklearn) |
| Threshold | 0.90 |

**Current state:** Inference running — test S2+S3 preprocessed (10M rows), S1 preprocessing (1.7M), then 35 chunks × 50k = full test set prediction

**Remaining tasks:**
- [ ] **Inference complete** → `output/matching_results.tsv` + `output/candidate_pairs.tsv`
- [ ] Run `python3 utils/validate_submission.py` → verify PASS
- [ ] Rerun with `--sample-size 50000` for stronger submission model
- [ ] Run `python3 package_submission.py --team-name "YourTeamName"` → final zip
- [ ] Final commit + push all changes

---

## 🚨 Golden Rules (Never Break These)

1. **NO external APIs.** No geocoding. No web lookups. Instant disqualification.
2. **Always `sep='\t'`** for every `.tsv` read/write.
3. **Every S1 entity must have exactly one row** in `matching_results.tsv` (even singletons).
4. **Singletons** → empty `matched_entity_ids` — score 1.0 each. Never omit.
5. **Never hard-code country logic** — France is 15% of test, absent from training.
6. **Run the validator** before every submission — exit code 0 = safe to submit.

---

## 🗂️ Final File Structure

```
amazon-ml-challenge/
│
├── dataset/
│   ├── train/                         → symlinks → ../student_resource/dataset/train/
│   └── test/                          → symlinks → ../student_resource/dataset/test/
│
├── src/business_entity_resolution/
│   ├── config.py                      TRAIN_SAMPLE_SIZE=50k, THRESHOLD=0.90, BLOCKING_TOP_K=15
│   ├── data/                          loader.py (sep='\t', dtype=str, keep_default_na=False)
│   ├── preprocessing/                 preprocess.py (VECTORIZED: str.replace, str.extract)
│   ├── blocking/                      exact_blocking.py (pandas merge + stopwords)
│   │                                  fuzzy_blocking.py (TF-IDF ANN, memory-safe)
│   │                                  candidate_generation.py (2M cap, format TSV)
│   ├── features/                      feature_builder.py (VECTORIZED, 14 features)
│   ├── models/                        model.py (auto-backend LGB→XGB→HistGBM→GBM)
│   │                                  train.py (positive-aware labels, class weights)
│   │                                  predict.py (threshold=0.90, singletons→"")
│   ├── evaluation/                    metrics.py (F0.5), validation.py (split)
│   ├── pipeline/                      train_pipeline.py (pos-aware sampling, sampled S1)
│   │                                  inference_pipeline.py (CHUNKED 50k, S2+S3 once)
│   └── main.py                        --mode [train|predict|all] --sample-size --threshold
│
├── output/
│   ├── matching_results.tsv           ← 1 row per S1 entity (leaderboard file)
│   └── candidate_pairs.tsv            ← 1 row per S1 entity (blocking output)
│
├── models_saved/
│   └── matching_model.pkl             ← Saved HistGBM model (gitignored)
│
├── utils/
│   └── validate_submission.py         ← Official validator (MUST pass before zip)
│
├── notebooks/
│   └── eda.py                         ← EDA script
│
├── package_submission.py              ← Validates + builds submission zip
├── README.md                          ← Full setup + run instructions
├── requirements.txt
├── PROJECT_PLAN.md                    ← This file
└── Documentation_template.md          ← Filled methodology (F_0.5=0.8642 recorded)
```

---

## 🧩 Module Dependency Map

```
main.py  (--mode all --sample-size 50000 --threshold 0.90)
 │
 ├── pipeline/train_pipeline.py
 │     ├── data/loader.py                reads TSVs (sep='\t')
 │     ├── preprocessing/preprocess.py   VECTORIZED name + address normalization
 │     ├── _preprocess_candidates_sampled()  POSITIVE-AWARE: guarantees true-match S2/S3
 │     ├── blocking/exact_blocking.py    pandas merge + stopword filter
 │     ├── blocking/fuzzy_blocking.py    TF-IDF char n-gram ANN (memory-safe)
 │     ├── features/feature_builder.py   VECTORIZED 14-dim similarity vector
 │     └── models/train.py               HistGBM / auto-backend, class_weight
 │
 └── pipeline/inference_pipeline.py     CHUNKED 50k S1 at a time
       ├── preprocessing/               S2+S3 preprocessed ONCE (full 10M), reused
       ├── blocking/                    per-chunk exact + TF-IDF blocking
       ├── features/                    per-chunk feature matrix
       ├── models/predict.py            threshold=0.90, singletons → ""
       └── output/                      matching_results.tsv + candidate_pairs.tsv
```

---

## 🎯 Scoring Notes

| Concept | Detail |
|---|---|
| **Metric** | Macro F_0.5 — precision weighted **2×** over recall |
| **Threshold** | `0.90` — only call "Match" when 90%+ confident |
| **False positives** | Catastrophic — avoid at all costs |
| **Singletons** | Always emit, always score 1.0 per entity |
| **France** | Test-only — TF-IDF handles it (no hard-coding needed) |
| **Validated F_0.5** | **0.8642** on 5k sample — expect similar or better on 50k |

---

## ⚡ Quick Run Commands

```bash
# 0. Setup (once)
cd amazon-ml-challenge/
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pip install -r requirements.txt

# 1. Fast test (5k sample, ~15 min end-to-end)
python3 src/business_entity_resolution/main.py --mode all --sample-size 5000 --threshold 0.90

# 2. Submission-quality run (50k sample, ~45-90 min)
python3 src/business_entity_resolution/main.py --mode all --sample-size 50000 --threshold 0.90

# 3. Predict-only with saved model (after training once)
python3 src/business_entity_resolution/main.py --mode predict --threshold 0.90

# 4. Validate output (MANDATORY — must pass before zip)
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 5. Package submission zip
python3 package_submission.py --team-name "YourTeamName"
```
