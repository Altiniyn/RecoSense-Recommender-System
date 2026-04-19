"""
recommenders/content_based.py
──────────────────────────────
Content-Based Engine — 2 methods:
  A. TF-IDF cosine similarity   (pure text match)
  B. Score-Weighted TF-IDF      (text weighted by community rating)

BUG FIXED:
  • sparse .mean(axis=0) returns np.matrix which sklearn ≥1.4 rejects.
    Wrapped in np.asarray() inside _user_vector so cosine_similarity
    always receives a plain ndarray.
  • Added missing self.df assignment in __init__ (was causing AttributeError
    in _user_vector and recommend on the very first call).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

METHODS = ["TF-IDF", "Weighted-TF-IDF"]


class ContentBasedEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df                          # ← FIX: was missing, caused AttributeError

        # ── aggregate one profile per product ───────────────────────────
        agg = (
            df.groupby("ProductId")
            .agg(
                text        = ("Text",  lambda x: " ".join(x.fillna("").astype(str))),
                avg_score   = ("Score", "mean"),
                num_reviews = ("Score", "count"),
            )
            .reset_index()
        )

        self.pids        = agg["ProductId"].tolist()
        self.avg_scores  = agg["avg_score"].values
        self.num_reviews = agg["num_reviews"].values

        # ── TF-IDF matrix ────────────────────────────────────────────────
        self._tfidf  = TfidfVectorizer(
            max_features=6000, stop_words="english", ngram_range=(1, 2)
        )
        self._matrix = self._tfidf.fit_transform(agg["text"])          # (P, V)  sparse

        # ── Score-weighted matrix ────────────────────────────────────────
        weights       = (agg["avg_score"] / 5.0).values.reshape(-1, 1)
        self._wmatrix = csr_matrix(self._matrix.multiply(weights))     # (P, V)  sparse

    # ── helpers ─────────────────────────────────────────────────────────

    def _idx(self, pid: str):
        try:   return self.pids.index(pid)
        except ValueError: return None

    def _user_vector(self, uid: str, mat):
        """
        Build a mean profile vector from the user's highly-rated products.

        BUG FIX: scipy sparse .mean(axis=0) returns np.matrix, which
        sklearn ≥ 1.4 refuses in cosine_similarity.  np.asarray() converts
        it to a plain (1, V) ndarray — zero semantic change.
        """
        liked = (
            self.df[(self.df["UserId"] == uid) & (self.df["Score"] >= 4)]["ProductId"]
            .tolist()
        )
        if not liked:
            liked = self.df[self.df["UserId"] == uid]["ProductId"].tolist()

        indices = [self._idx(p) for p in liked if self._idx(p) is not None]
        if not indices:
            return None

        # ← FIX: np.asarray converts np.matrix → ndarray
        return np.asarray(mat[indices].mean(axis=0))

    def _explain(self, method: str, pid: str, sim: float, avg: float) -> str:
        if method == "TF-IDF":
            return (
                f"Recommended because '{pid}' review content closely matches products "
                f"you rated highly (text similarity = {sim:.3f}, community avg = {avg:.1f} / 5)."
            )
        return (
            f"Score-weighted match: the highly-rated aspects of your taste profile "
            f"align strongly with '{pid}' (weighted similarity = {sim:.3f}, avg = {avg:.1f} / 5)."
        )

    # ── public API ───────────────────────────────────────────────────────

    def recommend(self, uid: str, method: str = "TF-IDF", n: int = 10) -> list[dict]:
        """
        Returns a list of dicts:
            {ProductId, SimilarityScore, AvgCommunityScore, Explanation}
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")

        mat      = self._matrix if method == "TF-IDF" else self._wmatrix
        user_vec = self._user_vector(uid, mat)
        if user_vec is None:
            return []

        rated = set(self.df[self.df["UserId"] == uid]["ProductId"])

        # user_vec is already (1, V) ndarray — cosine_similarity is happy
        sims  = cosine_similarity(user_vec, mat)[0]

        results = []
        for idx in np.argsort(sims)[::-1]:
            pid = self.pids[idx]
            if pid in rated:
                continue
            results.append({
                "ProductId":         pid,
                "SimilarityScore":   round(float(sims[idx]), 4),
                "AvgCommunityScore": round(float(self.avg_scores[idx]), 2),
                "NumReviews":        int(self.num_reviews[idx]),
                "Explanation":       self._explain(method, pid, sims[idx], self.avg_scores[idx]),
                "Method":            method,
                "Approach":          "Content-Based",
            })
            if len(results) == n:
                break
        return results

    def get_product_list(self) -> list[str]:
        return self.pids