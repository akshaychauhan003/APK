"""
Exploratory Data Analysis (EDA) Script
=======================================
Run this before coding the pipeline to understand noise patterns.

Usage (from amazon-ml-challenge/ directory):
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python3 notebooks/eda.py

This script reads TRAINING data ONLY — no test data is touched.
It produces console output. Redirect to a file if needed:
    python3 notebooks/eda.py > eda_output.txt 2>&1
"""

import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import re

from business_entity_resolution.config import (
    TRAIN_S1_PATH,
    TRAIN_S2_PATH,
    TRAIN_S3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    ENTITY_ID_COL,
    NAME_COL,
    ADDRESS_COL,
    COUNTRY_COL,
    GROUND_TRUTH_S1_COL,
    GROUND_TRUTH_MATCHED_COL,
)

SEP = "\t"

print("=" * 70)
print("AMAZON ML CHALLENGE — EDA REPORT")
print("=" * 70)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n[1] Loading datasets (sep='\\t') ...")
df_s1 = pd.read_csv(TRAIN_S1_PATH, sep=SEP, dtype=str, keep_default_na=False)
df_s2 = pd.read_csv(TRAIN_S2_PATH, sep=SEP, dtype=str, keep_default_na=False)
df_s3 = pd.read_csv(TRAIN_S3_PATH, sep=SEP, dtype=str, keep_default_na=False)
df_gt = pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep=SEP, dtype=str, keep_default_na=False)

print(f"  Source 1: {len(df_s1):>10,} rows   Columns: {list(df_s1.columns)}")
print(f"  Source 2: {len(df_s2):>10,} rows   Columns: {list(df_s2.columns)}")
print(f"  Source 3: {len(df_s3):>10,} rows   Columns: {list(df_s3.columns)}")
print(f"  GT file : {len(df_gt):>10,} rows   Columns: {list(df_gt.columns)}")

# ── 2. Country Distribution ───────────────────────────────────────────────────
print("\n[2] Country Distribution:")
for name, df in [("S1", df_s1), ("S2", df_s2), ("S3", df_s3)]:
    counts = df[COUNTRY_COL].value_counts()
    print(f"  {name}: {dict(counts)}")

# ── 3. Missing Data ───────────────────────────────────────────────────────────
print("\n[3] Missing / Empty Values:")
for name, df in [("S1", df_s1), ("S2", df_s2), ("S3", df_s3)]:
    missing_name = (df[NAME_COL].str.strip() == "").sum()
    missing_addr = (df[ADDRESS_COL].str.strip() == "").sum()
    print(f"  {name}: empty name={missing_name:,}  empty address={missing_addr:,}")

# ── 4. Ground Truth Match Statistics ─────────────────────────────────────────
print("\n[4] Ground Truth Match Statistics:")
df_gt["match_count"] = df_gt[GROUND_TRUTH_MATCHED_COL].apply(
    lambda x: len([i for i in x.split(",") if i.strip()]) if x.strip() else 0
)
singletons = (df_gt["match_count"] == 0).sum()
matched = (df_gt["match_count"] > 0).sum()
max_matches = df_gt["match_count"].max()
avg_matches = df_gt[df_gt["match_count"] > 0]["match_count"].mean()
print(f"  Total S1 entities in GT : {len(df_gt):,}")
print(f"  Singletons (no matches) : {singletons:,} ({100*singletons/len(df_gt):.1f}%)")
print(f"  Has at least one match  : {matched:,} ({100*matched/len(df_gt):.1f}%)")
print(f"  Max matches per entity  : {max_matches}")
print(f"  Avg matches (when >0)   : {avg_matches:.2f}")

# ── 5. Name Length Distribution ───────────────────────────────────────────────
print("\n[5] Business Name Length (characters):")
for name, df in [("S1", df_s1), ("S2", df_s2), ("S3", df_s3)]:
    lengths = df[NAME_COL].str.len()
    print(f"  {name}: min={lengths.min()}, median={lengths.median():.0f}, max={lengths.max()}, mean={lengths.mean():.1f}")

# ── 6. Noise Pattern Samples ──────────────────────────────────────────────────
print("\n[6] Sample Business Names from each source (first 5 per country):")
for country in df_s1[COUNTRY_COL].unique()[:3]:
    print(f"\n  Country: {country}")
    sample_s1 = df_s1[df_s1[COUNTRY_COL] == country][NAME_COL].head(3).tolist()
    sample_s2 = df_s2[df_s2[COUNTRY_COL] == country][NAME_COL].head(3).tolist()
    print(f"    S1: {sample_s1}")
    print(f"    S2: {sample_s2}")

# ── 7. Address Postal Code Coverage ──────────────────────────────────────────
print("\n[7] Postal/PIN Code Coverage:")
postal_re = re.compile(r"\b\d{5,6}\b")
for name, df in [("S1", df_s1), ("S2", df_s2), ("S3", df_s3)]:
    has_postal = df[ADDRESS_COL].apply(lambda x: bool(postal_re.search(x))).sum()
    pct = 100 * has_postal / len(df)
    print(f"  {name}: {has_postal:,} / {len(df):,} rows have a 5-6 digit postal code ({pct:.1f}%)")

# ── 8. Legal Suffix Prevalence ────────────────────────────────────────────────
print("\n[8] Legal Suffix Prevalence in Business Names:")
suffixes = {
    "corp/corporation": r"\bcorp(oration)?\b",
    "pvt/private": r"\bpvt\b|\bprivate\b",
    "ltd/limited": r"\bltd\b|\blimited\b",
    "llc": r"\bllc\b",
    "inc": r"\binc\b",
}
for name, df in [("S1", df_s1), ("S2", df_s2)]:
    print(f"  {name}:")
    for suffix_name, pattern in suffixes.items():
        count = df[NAME_COL].str.lower().str.contains(pattern, regex=True).sum()
        print(f"    {suffix_name:20s}: {count:,}")

print("\n" + "=" * 70)
print("EDA COMPLETE — Review the patterns above before building your pipeline.")
print("=" * 70)
