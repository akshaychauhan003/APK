# ML Challenge 2026: Business Entity Resolution

**Team Name:** APK
**Team Members:** Akshay Singh Chauhan
**Submission Date:** September 25, 2026

---

## 1. Summary

Three-stage pipeline: blocking → feature engineering → gradient-boosted classifier.

We use a 0.90 decision threshold tuned for F_0.5 (precision weighted 2× over recall), so we only merge entities when the model is highly confident. Singletons (no match found) get blank `matched_entity_ids`, scoring 1.0 each.

---

## 2. Methodology

### 2.1 Noise Patterns

From EDA on the training data:
- Legal suffix variations: "Corp" / "Corporation" / "Incorporated" / "Pvt" / "Limited"
- Address abbreviations: "St" → "Street", "Rd" → "Road", "Blvd" → "Boulevard"
- Missing postal codes (common in Indian addresses); French addresses reorder them
- Mixed casing everywhere
- France only appears in test data → can't hardcode country-specific logic

### 2.2 Approach

**Type:** Hybrid blocking + binary classifier
**Key idea:** Multi-key blocking (exact + TF-IDF char n-gram per country) to get candidates, then a 14-feature similarity vector fed into LightGBM with phonetic matching for typo robustness.

---

## 3. Blocking (Candidate Generation)

Three strategies, unioned:

1. **Name block:** `(country, first_word_of_clean_name)` — groups entities sharing the same first significant word. Stopwords ("the", "national", etc.) and single-letter first words are filtered to prevent cartesian explosions.

2. **Postal block:** `(country, postal_code)` — extracted 5–6 digit codes via regex `\b\d{5,6}\b`.

3. **TF-IDF block:** Per-country TF-IDF on `clean_name + clean_address` with char 2–4 grams, cosine-similarity nearest neighbors retrieving top-15 per entity.

Why three strategies: a true match missed by postal (missing PIN) can still be caught by name or TF-IDF.

---

## 4. Features & Model

### Features (14)

| Feature | Description |
|---|---|
| `name_jaccard` | Token-level Jaccard between cleaned names |
| `name_first_word` | 1 if first tokens match, else 0 |
| `name_fuzz` | RapidFuzz edit distance ratio |
| `name_token_sort` | Word-order-invariant fuzz |
| `name_token_set` | Subset-tolerant fuzz |
| `name_len_diff` | Normalized length difference |
| `name_soundex` | 1 if Soundex codes match |
| `name_metaphone` | 1 if Metaphone codes match |
| `addr_jaccard` | Token-level Jaccard on cleaned addresses |
| `postal_match` | 1.0 exact, 0.5 if one side missing, 0.0 mismatch |
| `country_match` | Same country flag |
| `digit_jaccard` | Jaccard over numeric tokens in address |
| `addr_fuzz` | Edit distance on full address |
| `addr_token_set` | Token set ratio on address |

### Model

LightGBM (`n_estimators=300, lr=0.05, max_depth=7`) with `scale_pos_weight` for the ~85:1 class imbalance. Falls back to XGBoost or sklearn HistGBM if LightGBM isn't installed.

Threshold = 0.90 — accepts missing some matches rather than risking incorrect merges.

---

## 5. Results

- **F_0.5 (macro) = 0.8642** on 5k S1 sample with 0.90 threshold (HistGBM backend)
  - Precision = 0.8776, Recall = 0.8653
  - 12,504 positives / 1,069,423 negatives in training

**Error patterns:**
- False positives: generic first words (e.g. "Raj Enterprises" ≠ "Raj Services")
- False negatives: very short names + no postal code where blocking can't find the candidate

**Key insight:** blocking recall is the real bottleneck — the classifier can only match entities that made it into the candidate set. We guarantee all true-match IDs are in the training candidate pool to avoid training on unrealistically easy negatives-only data.

---

## 6. Conclusion

The pipeline handles US, Indian, and French business records with different formatting, missing data, and abbreviation patterns. Main limitation is blocking recall for very short abbreviated names. Future work: MinHash blocking, BERT name encoder for multilingual.

---

## Appendix: Running the Code

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src

python3 src/business_entity_resolution/main.py --mode all --threshold 0.90

python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Output files:**
- `output/matching_results.tsv` — one row per S1 entity
- `output/candidate_pairs.tsv` — candidate list from blocking

All data is TSV (`sep='\t'`). No external APIs or internet lookups.
