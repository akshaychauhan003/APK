import logging
from pathlib import Path

import pandas as pd

from ..data import load_test
from ..preprocessing import preprocess
from ..blocking import generate_candidates, format_candidate_tsv, precompute_blocking_keys
from ..features import build_features
from ..models import predict
from ..config import ID_COL, S1_ID_COL, MATCHED_COL, THRESHOLD, OUTPUT_MATCHING, OUTPUT_CANDIDATES

log = logging.getLogger(__name__)

CHUNK_SIZE = 50_000  # S1 entities per inference chunk to keep memory bounded


def run_inference(
    threshold: float = THRESHOLD,
    out_matching: Path = OUTPUT_MATCHING,
    out_candidates: Path = OUTPUT_CANDIDATES,
):
    log.info("loading test data...")
    df_s1, df_s2, df_s3 = load_test()
    log.info(f"  S1={len(df_s1):,}  S2={len(df_s2):,}  S3={len(df_s3):,}")

    # 1. Save all_s1_ids as a proper copy before preprocessing
    all_s1_ids = df_s1[ID_COL].copy()

    log.info("preprocessing S2+S3 (done once)...")
    df_s2 = preprocess(df_s2)
    df_s3 = preprocess(df_s3)

    # Pre-compute deduplicated candidate DataFrame ONCE (saves ~5 min per chunk)
    log.info("deduplicating S2+S3 candidates (done once)...")
    df_s2_s3 = pd.concat([df_s2, df_s3], ignore_index=True)
    df_cand_deduped = df_s2_s3.drop_duplicates(subset=[ID_COL])
    log.info(f"  candidates: {len(df_cand_deduped):,} unique (from {len(df_s2_s3):,})")

    # Pre-compute candidate blocking keys ONCE (saves ~5 min per chunk)
    log.info("pre-computing candidate blocking keys (done once)...")
    precomputed_cand, gram_counts = precompute_blocking_keys(df_cand_deduped, role="cand")
    log.info(f"  candidate blocking keys: {len(precomputed_cand):,} records, {len(gram_counts):,} 3-grams")

    log.info("preprocessing S1...")
    df_s1 = preprocess(df_s1)

    n_chunks = (len(df_s1) + CHUNK_SIZE - 1) // CHUNK_SIZE
    log.info(f"chunked inference: {n_chunks} chunks of {CHUNK_SIZE:,}")

    all_pairs    = []
    all_matching = []

    for i in range(n_chunks):
        lo = i * CHUNK_SIZE
        hi = min(lo + CHUNK_SIZE, len(df_s1))
        chunk = df_s1.iloc[lo:hi].copy()
        chunk_ids = chunk[ID_COL]
        log.info(f"  chunk {i+1}/{n_chunks}: rows {lo:,}–{hi:,}")

        # 5. Error handling so crashes fallback to empty rows
        try:
            pairs = generate_candidates(
                chunk, df_s2, df_s3,
                precomputed_cand=precomputed_cand,
                gram_counts=gram_counts,
                df_cand_deduped=df_cand_deduped,
            )
            all_pairs.append(pairs)

            if pairs.empty:
                all_matching.append(pd.DataFrame({S1_ID_COL: chunk_ids, MATCHED_COL: ""}))
                continue

            feats = build_features(pairs, chunk, df_s2_s3)
            preds = predict(feats, all_s1_ids=chunk_ids, threshold=threshold)
            all_matching.append(preds)
        except Exception as e:
            log.error(f"Error in chunk {i+1}: {e}")
            import traceback
            log.error(traceback.format_exc())
            all_matching.append(pd.DataFrame({S1_ID_COL: chunk_ids, MATCHED_COL: ""}))

    log.info("combining results...")
    if all_pairs:
        df_pairs = pd.concat(all_pairs, ignore_index=True)
    else:
        df_pairs = pd.DataFrame()
        
    df_results = pd.concat(all_matching, ignore_index=True)
    df_results = df_results.drop_duplicates(subset=[S1_ID_COL], keep="first")

    # 2. Final LEFT JOIN from the complete S1 ID list
    s1_df = pd.DataFrame({S1_ID_COL: all_s1_ids})
    df_results = s1_df.merge(df_results, on=S1_ID_COL, how="left")
    df_results[MATCHED_COL] = df_results[MATCHED_COL].fillna("")

    # 3. Add assertion to verify completeness
    assert len(df_results) == len(all_s1_ids), f"Missing rows: expected {len(all_s1_ids)}, got {len(df_results)}"

    log.info("saving candidate_pairs.tsv...")
    df_cand_tsv = format_candidate_tsv(df_pairs, all_s1_ids)
    df_cand_tsv.to_csv(out_candidates, sep="\t", index=False)
    log.info(f"  {out_candidates}  ({len(df_cand_tsv):,} rows)")

    log.info("saving matching_results.tsv...")
    df_results.to_csv(out_matching, sep="\t", index=False)
    log.info(f"  {out_matching}  ({len(df_results):,} rows)")

    matched = (df_results[MATCHED_COL] != "").sum()
    log.info(f"done — {matched:,} matched, {len(df_results)-matched:,} singletons")
    return df_results, df_cand_tsv
