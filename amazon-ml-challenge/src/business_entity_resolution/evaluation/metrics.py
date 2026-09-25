from ..config import S1_ID_COL, MATCHED_COL, F_BETA


def f_beta(precision: float, recall: float, beta: float = F_BETA) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def evaluate(df_pred, df_gt) -> dict:
    """Compute macro precision, recall, F_0.5 comparing predictions to ground truth.
    
    Handles singletons: correctly predicting no-match scores 1.0 per the challenge spec.
    """
    def parse(s):
        s = str(s).strip()
        return set(s.split(",")) if s and s.lower() != "nan" else set()

    # build lookup dicts without iterrows — use pandas indexing
    pred_map = df_pred.set_index(S1_ID_COL)[MATCHED_COL].apply(parse).to_dict()
    gt_map   = df_gt.set_index(S1_ID_COL)[MATCHED_COL].apply(parse).to_dict()

    precs, recs, f05s = [], [], []
    for s1_id, actual in gt_map.items():
        pred = pred_map.get(s1_id, set())
        tp = len(pred & actual)

        if not pred and not actual:
            p = r = 1.0
        elif not pred:
            p, r = 1.0, 0.0
        elif not actual:
            p, r = 0.0, 1.0
        else:
            p = tp / len(pred)
            r = tp / len(actual)

        precs.append(p)
        recs.append(r)
        f05s.append(f_beta(p, r))

    n = len(f05s) or 1
    return {
        "macro_precision": sum(precs) / n,
        "macro_recall":    sum(recs) / n,
        "macro_f0.5":      sum(f05s) / n,
    }
