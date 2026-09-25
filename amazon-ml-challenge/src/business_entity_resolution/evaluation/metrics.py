"""
Evaluation metric utilities calculating precision, recall, and Macro F0.5 score.
"""

from typing import Dict, Set
import pandas as pd
from ..config import GROUND_TRUTH_S1_COL, GROUND_TRUTH_MATCHED_COL, F_BETA


def calculate_f_beta(precision: float, recall: float, beta: float = F_BETA) -> float:
    """Calculate F-beta score given precision and recall."""
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return numerator / denominator if denominator > 0 else 0.0


def evaluate_predictions(df_pred: pd.DataFrame, df_gt: pd.DataFrame) -> Dict[str, float]:
    """Calculate macro precision, recall, and F0.5 score comparing predictions against ground truth."""
    pred_map: Dict[str, Set[str]] = {}
    for _, row in df_pred.iterrows():
        s1_id = row[GROUND_TRUTH_S1_COL]
        matched = set(row[GROUND_TRUTH_MATCHED_COL].split(",")) if row[GROUND_TRUTH_MATCHED_COL] else set()
        pred_map[s1_id] = matched

    gt_map: Dict[str, Set[str]] = {}
    for _, row in df_gt.iterrows():
        s1_id = row[GROUND_TRUTH_S1_COL]
        matched = set(row[GROUND_TRUTH_MATCHED_COL].split(",")) if row[GROUND_TRUTH_MATCHED_COL] else set()
        gt_map[s1_id] = matched

    precisions = []
    recalls = []
    f_scores = []

    for s1_id, actual_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())

        tp = len(pred_set.intersection(actual_set))
        fp = len(pred_set - actual_set)
        fn = len(actual_set - pred_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not actual_set and not pred_set else 0.0)
        r = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not actual_set and not pred_set else 0.0)
        f05 = calculate_f_beta(p, r, beta=F_BETA)

        precisions.append(p)
        recalls.append(r)
        f_scores.append(f05)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f05 = sum(f_scores) / len(f_scores) if f_scores else 0.0

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f0.5": macro_f05,
    }
