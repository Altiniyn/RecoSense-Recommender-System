"""
recommenders/knowledge_based.py
────────────────────────────────
Knowledge-Based Engine — constraint + ranking approach.

User specifies:
  • min_score       – minimum average community rating
  • keyword         – must appear in review text
  • min_reviews     – minimum number of reviews (popularity floor)
  • max_reviews     – maximum number of reviews (niche ceiling)
  • ranking         – "score" | "helpfulness" | "popularity"

Two main ranking strategies (= 2 sub-methods):
  A. Score-ranked        – highest avg community rating first
  B. Helpfulness-ranked  – highest avg helpfulness ratio first
  C. Popularity-ranked   – most-reviewed first (bonus 3rd strategy)
"""

import numpy as np
import pandas as pd

RANKING_OPTIONS = ["score", "helpfulness", "popularity"]


class KnowledgeBasedEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # compute helpfulness ratio
        denom = df["HelpfulnessDenominator"].replace(0, np.nan)
        df    = df.copy()
        df["helpfulness_ratio"] = df["HelpfulnessNumerator"] / denom

        self.catalog = (
            df.groupby("ProductId")
            .agg(
                avg_score          = ("Score",             "mean"),
                num_reviews        = ("Score",             "count"),
                avg_helpfulness    = ("helpfulness_ratio", "mean"),
                combined_text      = ("Text",
                                      lambda x: " ".join(x.fillna("").astype(str).str.lower())),
            )
            .reset_index()
        )
        self.catalog["avg_helpfulness"] = self.catalog["avg_helpfulness"].fillna(0)

    # ── public API ───────────────────────────────────────────────────────

    def recommend(
        self,
        uid:         str,
        min_score:   float = 4.0,
        keyword:     str   = "",
        min_reviews: int   = 5,
        max_reviews: int   = 999_999,
        ranking:     str   = "score",
        n:           int   = 10,
    ) -> list[dict]:
        """
        Returns a list of dicts:
            {ProductId, AvgScore, NumReviews, AvgHelpfulness, Explanation}
        """
        rated = set(self.df[self.df["UserId"] == uid]["ProductId"])
        cat   = self.catalog.copy()

        # ── apply hard constraints ───────────────────────────────────────
        cat = cat[~cat["ProductId"].isin(rated)]
        cat = cat[cat["avg_score"]   >= min_score]
        cat = cat[cat["num_reviews"].between(min_reviews, max_reviews)]

        if keyword.strip():
            kw  = keyword.strip().lower()
            cat = cat[cat["combined_text"].str.contains(kw, na=False, regex=False)]

        if cat.empty:
            return []

        # ── rank ─────────────────────────────────────────────────────────
        sort_map = {
            "score":       "avg_score",
            "helpfulness": "avg_helpfulness",
            "popularity":  "num_reviews",
        }
        sort_col = sort_map.get(ranking, "avg_score")
        cat = cat.sort_values(sort_col, ascending=False).head(n)

        results = []
        for _, row in cat.iterrows():
            kw_part = f", keyword='{keyword}'" if keyword.strip() else ""
            results.append({
                "ProductId":      row["ProductId"],
                "AvgScore":       round(row["avg_score"], 2),
                "NumReviews":     int(row["num_reviews"]),
                "AvgHelpfulness": round(row["avg_helpfulness"], 3),
                "Explanation": (
                    f"Recommended based on your constraints "
                    f"(min rating ≥ {min_score}{kw_part}): "
                    f"avg = {row['avg_score']:.1f} / 5, "
                    f"{int(row['num_reviews'])} reviews, "
                    f"ranked by {ranking}."
                ),
                "Method":   f"{ranking.capitalize()}-ranked",
                "Approach": "Knowledge-Based",
            })
        return results

    def get_score_range(self) -> tuple[float, float]:
        return (
            float(self.catalog["avg_score"].min()),
            float(self.catalog["avg_score"].max()),
        )

    def get_review_range(self) -> tuple[int, int]:
        return (
            int(self.catalog["num_reviews"].min()),
            int(self.catalog["num_reviews"].max()),
        )
