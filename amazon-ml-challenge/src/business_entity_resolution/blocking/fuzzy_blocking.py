"""
Fuzzy candidate generation using TF-IDF character n-gram cosine similarity.
"""

from typing import Set, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..config import ENTITY_ID_COL, COUNTRY_COL, BLOCKING_TOP_K


def generate_tfidf_candidates(
    df_s1: pd.DataFrame,
    df_candidates: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
) -> Set[Tuple[str, str]]:
    """
    Generate candidate pairs per country using TF-IDF character n-grams and Nearest Neighbors.
    """
    pairs: Set[Tuple[str, str]] = set()

    # Process per country partition
    countries = set(df_s1[COUNTRY_COL].unique()).union(set(df_candidates[COUNTRY_COL].unique()))

    for country in countries:
        sub_s1 = df_s1[df_s1[COUNTRY_COL] == country].copy()
        sub_cand = df_candidates[df_candidates[COUNTRY_COL] == country].copy()

        if sub_s1.empty or sub_cand.empty:
            continue

        s1_texts = sub_s1["clean_name"] + " " + sub_s1["clean_address"]
        cand_texts = sub_cand["clean_name"] + " " + sub_cand["clean_address"]

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        try:
            cand_matrix = vectorizer.fit_transform(cand_texts)
            s1_matrix = vectorizer.transform(s1_texts)
        except ValueError:
            continue

        n_neighbors = min(top_k, cand_matrix.shape[0])
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        nn.fit(cand_matrix)

        distances, indices = nn.kneighbors(s1_matrix)

        s1_ids = sub_s1[ENTITY_ID_COL].values
        cand_ids = sub_cand[ENTITY_ID_COL].values

        for i, neighbor_indices in enumerate(indices):
            s1_id = s1_ids[i]
            for idx in neighbor_indices:
                pairs.add((s1_id, cand_ids[idx]))

    return pairs
