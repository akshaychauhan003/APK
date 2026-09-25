"""
Fuzzy candidate generation using TF-IDF character n-gram cosine similarity.

For large country partitions (millions of candidates), we use a random
sub-sample of candidates to keep memory usage bounded. The exact blocking
layer handles high-recall cases; the TF-IDF layer adds fuzzy coverage.
"""

import logging
from typing import Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..config import ENTITY_ID_COL, COUNTRY_COL, BLOCKING_TOP_K

logger = logging.getLogger(__name__)

# Max candidates per country to fit in TF-IDF. For countries with more records,
# we sub-sample. This bounds memory to roughly 200k × vocab_size floats.
MAX_CAND_PER_COUNTRY = 200_000


def generate_tfidf_candidates(
    df_s1: pd.DataFrame,
    df_candidates: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
) -> Set[Tuple[str, str]]:
    """
    Generate candidate pairs per country using TF-IDF character n-grams and ANN.

    Country-partitioned to avoid cross-country false positives and to keep
    matrix sizes manageable. Large partitions are sub-sampled on the candidate
    side (recall is supplemented by the exact blocking layer).
    """
    pairs: Set[Tuple[str, str]] = set()

    # Union of all country values across S1 and candidate pool
    countries = (
        set(df_s1[COUNTRY_COL].dropna().unique())
        | set(df_candidates[COUNTRY_COL].dropna().unique())
    )

    for country in sorted(countries):
        sub_s1 = df_s1[df_s1[COUNTRY_COL] == country]
        sub_cand = df_candidates[df_candidates[COUNTRY_COL] == country]

        if sub_s1.empty or sub_cand.empty:
            continue

        # Sub-sample large candidate partitions to bound memory usage
        if len(sub_cand) > MAX_CAND_PER_COUNTRY:
            logger.info(
                f"  TF-IDF [{country}]: {len(sub_cand):,} candidates → "
                f"sub-sampling to {MAX_CAND_PER_COUNTRY:,}"
            )
            sub_cand = sub_cand.sample(
                n=MAX_CAND_PER_COUNTRY, random_state=42
            )
        else:
            logger.info(
                f"  TF-IDF [{country}]: {len(sub_s1):,} S1 × {len(sub_cand):,} candidates"
            )

        s1_texts = (sub_s1["clean_name"] + " " + sub_s1["clean_address"]).fillna("").values
        cand_texts = (sub_cand["clean_name"] + " " + sub_cand["clean_address"]).fillna("").values

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=50_000,   # cap vocabulary to control memory
            sublinear_tf=True,
        )
        try:
            cand_matrix = vectorizer.fit_transform(cand_texts)
            s1_matrix = vectorizer.transform(s1_texts)
        except ValueError as e:
            logger.warning(f"  TF-IDF [{country}]: skipped ({e})")
            continue

        n_neighbors = min(top_k, cand_matrix.shape[0])
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(cand_matrix)

        # Process S1 in batches to avoid building a huge distance matrix at once
        s1_ids = sub_s1[ENTITY_ID_COL].values
        cand_ids = sub_cand[ENTITY_ID_COL].values

        BATCH = 5_000
        for batch_start in range(0, len(s1_ids), BATCH):
            batch_end = min(batch_start + BATCH, len(s1_ids))
            _, indices = nn.kneighbors(s1_matrix[batch_start:batch_end])
            for i, neighbor_indices in enumerate(indices):
                s1_id = s1_ids[batch_start + i]
                for idx in neighbor_indices:
                    pairs.add((s1_id, cand_ids[idx]))

        logger.info(f"  TF-IDF [{country}]: {len(pairs):,} pairs accumulated so far")

    return pairs
