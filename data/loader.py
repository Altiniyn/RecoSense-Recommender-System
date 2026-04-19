"""
data/loader.py
──────────────
Loads and cleans the Amazon Fine Food Reviews CSV.

Expected columns (from Kaggle):
    Id, ProductId, UserId, ProfileName,
    HelpfulnessNumerator, HelpfulnessDenominator,
    Score, Time, Summary, Text
"""

import pandas as pd
import numpy as np


def load_reviews(csv_path: str, sample_size: int = 60_000) -> pd.DataFrame:
    """
    Load Reviews.csv, clean it, and return a ready-to-use DataFrame.

    Parameters
    ----------
    csv_path   : path to Reviews.csv
    sample_size: max rows to keep (for speed; use None to load all)

    Returns
    -------
    pd.DataFrame with columns:
        UserId, ProductId, Score, Text, Summary,
        HelpfulnessNumerator, HelpfulnessDenominator
    """
    df = pd.read_csv(csv_path)

    # ── validate required columns ─────────────────────────────────────────
    required = {"ProductId", "UserId", "Score", "Text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ── clean ─────────────────────────────────────────────────────────────
    df = df.dropna(subset=["ProductId", "UserId", "Score", "Text"])
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").clip(1, 5)
    df = df.dropna(subset=["Score"])
    df["Score"] = df["Score"].astype(int)

    # fill optional columns
    for col in ["Summary", "ProfileName"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["HelpfulnessNumerator", "HelpfulnessDenominator"]:
        if col not in df.columns:
            df[col] = 0
    if "Time" not in df.columns:
        df["Time"] = 0

    df["HelpfulnessNumerator"]   = pd.to_numeric(df["HelpfulnessNumerator"],   errors="coerce").fillna(0).astype(int)
    df["HelpfulnessDenominator"] = pd.to_numeric(df["HelpfulnessDenominator"], errors="coerce").fillna(0).astype(int)

    # ── keep products & users with ≥5 interactions ────────────────────────
    for col in ["ProductId", "UserId"]:
        counts = df[col].value_counts()
        df = df[df[col].isin(counts[counts >= 5].index)]

    # ── sample ────────────────────────────────────────────────────────────
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # one rating per (user, product) pair – keep the latest
    df = df.sort_values("Time").drop_duplicates(
        subset=["UserId", "ProductId"], keep="last"
    ).reset_index(drop=True)

    return df[[
        "UserId", "ProductId", "Score", "Text", "Summary",
        "HelpfulnessNumerator", "HelpfulnessDenominator", "Time"
    ]]


def dataset_stats(df: pd.DataFrame) -> dict:
    """Return a quick summary dict for display."""
    return {
        "total_reviews": len(df),
        "unique_users":  df["UserId"].nunique(),
        "unique_products": df["ProductId"].nunique(),
        "avg_score":     round(df["Score"].mean(), 2),
        "sparsity":      round(
            1 - len(df) / (df["UserId"].nunique() * df["ProductId"].nunique()), 4
        ),
        "score_dist": df["Score"].value_counts().sort_index().to_dict(),
    }
