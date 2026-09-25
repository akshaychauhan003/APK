import logging
from pathlib import Path

import pandas as pd

from ..data import load_test
from ..preprocessing import preprocess
from ..blocking import generate_candidates, format_candidate_tsv
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

    log.info("preprocessing S2+S3 (done once)...")
    df_s2 = preprocess(df_s2)
    df_s3 = preprocess(df_s3)

    log.info("preprocessing S1...")
    df_s1 = preprocess(df_s1)
    all_s1_ids = df_s1[ID_COL]

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

        pairs = generate_candidates(chunk, df_s2, df_s3)
        all_pairs.append(pairs)

        if pairs.empty:
            all_matching.append(pd.DataFrame({S1_ID_COL: chunk_ids, MATCHED_COL: ""}))
            continue

        feats = build_features(pairs, chunk, pd.concat([df_s2, df_s3], ignore_index=True))
        preds = predict(feats, all_s1_ids=chunk_ids, threshold=threshold)
        all_matching.append(preds)

    log.info("combining results...")
    df_pairs   = pd.concat(all_pairs, ignore_index=True)
    df_results = pd.concat(all_matching, ignore_index=True)
    df_results = df_results.drop_duplicates(subset=[S1_ID_COL], keep="first")

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
