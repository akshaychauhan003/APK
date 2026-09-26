import logging
from typing import Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..config import ID_COL, COUNTRY_COL, BLOCKING_TOP_K

log = logging.getLogger(__name__)

MAX_CAND_PER_COUNTRY = 200_000


def generate_tfidf_candidates(
    df_s1: pd.DataFrame,
    df_cand: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
) -> Set[Tuple[str, str]]:
    """TF-IDF char n-gram ANN blocking, run per country to avoid cross-country noise."""
    pairs: Set[Tuple[str, str]] = set()

    countries = (
        set(df_s1[COUNTRY_COL].dropna().unique())
        | set(df_cand[COUNTRY_COL].dropna().unique())
    )

    for country in sorted(countries):
        sub_s1   = df_s1[df_s1[COUNTRY_COL] == country]
        sub_cand = df_cand[df_cand[COUNTRY_COL] == country]

        if sub_s1.empty or sub_cand.empty:
            continue

        if len(sub_cand) > MAX_CAND_PER_COUNTRY:
            log.info(f"  tfidf [{country}]: {len(sub_cand):,} candidates — subsampling to {MAX_CAND_PER_COUNTRY:,}")
            sub_cand = sub_cand.sample(n=MAX_CAND_PER_COUNTRY, random_state=42)
        else:
            log.info(f"  tfidf [{country}]: {len(sub_s1):,} S1 x {len(sub_cand):,} candidates")

        s1_texts   = (sub_s1["clean_name"]   + " " + sub_s1["clean_addr"] + " " + sub_s1["postal"]).fillna("").values
        cand_texts = (sub_cand["clean_name"] + " " + sub_cand["clean_addr"] + " " + sub_cand["postal"]).fillna("").values

        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            min_df=1, max_features=30_000, sublinear_tf=True,
        )
        try:
            cand_mat = vec.fit_transform(cand_texts)
            s1_mat   = vec.transform(s1_texts)
        except ValueError as e:
            log.warning(f"  tfidf [{country}]: skipped ({e})")
            continue

        k = min(top_k, cand_mat.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(cand_mat)

        s1_ids   = sub_s1[ID_COL].values
        cand_ids = sub_cand[ID_COL].values

        BATCH = 5_000
        for start in range(0, len(s1_ids), BATCH):
            end = min(start + BATCH, len(s1_ids))
            _, idxs = nn.kneighbors(s1_mat[start:end])
            for i, nbrs in enumerate(idxs):
                sid = s1_ids[start + i]
                for idx in nbrs:
                    pairs.add((sid, cand_ids[idx]))

        log.info(f"  tfidf [{country}]: {len(pairs):,} pairs so far")

    return pairs
