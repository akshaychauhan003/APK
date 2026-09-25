import pandas as pd
from sklearn.model_selection import train_test_split
from ..config import RANDOM_STATE, ID_COL, S1_ID_COL


def split_train_val(df_s1: pd.DataFrame, df_gt: pd.DataFrame, val_size: float = 0.20):
    ids = df_s1[ID_COL].unique()
    train_ids, val_ids = train_test_split(ids, test_size=val_size, random_state=RANDOM_STATE)

    return (
        df_s1[df_s1[ID_COL].isin(train_ids)].copy(),
        df_s1[df_s1[ID_COL].isin(val_ids)].copy(),
        df_gt[df_gt[S1_ID_COL].isin(train_ids)].copy(),
        df_gt[df_gt[S1_ID_COL].isin(val_ids)].copy(),
    )
