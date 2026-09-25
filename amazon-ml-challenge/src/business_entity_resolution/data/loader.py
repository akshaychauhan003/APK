import pandas as pd
from pathlib import Path
from ..config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, TEST_S1, TEST_S2, TEST_S3


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_train():
    return read_tsv(TRAIN_S1), read_tsv(TRAIN_S2), read_tsv(TRAIN_S3), read_tsv(TRAIN_GT)


def load_test():
    return read_tsv(TEST_S1), read_tsv(TEST_S2), read_tsv(TEST_S3)
