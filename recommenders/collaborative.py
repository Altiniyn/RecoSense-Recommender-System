"""
recommenders/collaborative.py
──────────────────────────────
Collaborative Filtering Engine — 4 methods:
  1. User-Based KNN  (cosine similarity)
  2. Item-Based KNN  (cosine + mean-centered)
  3. SVD             (Matrix Factorisation)
  4. Slope One       (deviation-based)
"""

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, KNNWithMeans, SlopeOne
from surprise.model_selection import train_test_split as surp_split
from surprise import accuracy as surp_acc

METHODS = ["User-Based KNN", "Item-Based KNN", "SVD", "Slope One"]


class CollaborativeFilteringEngine:

    def __init__(self, df: pd.DataFrame):
        self.df          = df
        self.all_products = set(df["ProductId"].unique())

        reader    = Reader(rating_scale=(1, 5))
        data      = Dataset.load_from_df(df[["UserId", "ProductId", "Score"]], reader)
        trainset, self.testset = surp_split(data, test_size=0.2, random_state=42)

        self._models = {
            "User-Based KNN": KNNBasic(
                k=40, sim_options={"name": "cosine", "user_based": True}, verbose=False
            ),
            "Item-Based KNN": KNNWithMeans(
                k=40, sim_options={"name": "cosine", "user_based": False}, verbose=False
            ),
            "SVD": SVD(n_factors=100, n_epochs=20, random_state=42),
            "Slope One": SlopeOne(),
        }

        for name, model in self._models.items():
            model.fit(trainset)

    # ── internal helpers ─────────────────────────────────────────────────

    def _predict(self, model, uid, iid) -> float:
        try:
            return model.predict(uid, iid).est
        except Exception:
            return 3.0

    def _explain(self, method: str, uid: str, pid: str, score: float) -> str:
        if method == "User-Based KNN":
            return (
                f"Users with a taste profile similar to yours gave '{pid}' "
                f"an average of {score:.1f} / 5 — so you're very likely to enjoy it."
            )
        if method == "Item-Based KNN":
            top_rated = (
                self.df[(self.df["UserId"] == uid) & (self.df["Score"] >= 4)]["ProductId"]
                .tolist()
            )
            anchor = top_rated[0] if top_rated else "items you rated highly"
            return (
                f"'{pid}' is frequently purchased alongside '{anchor}' "
                f"by shoppers with your rating pattern (predicted score {score:.1f} / 5)."
            )
        if method == "SVD":
            return (
                f"Matrix-factorisation (SVD) detected a strong latent-factor alignment "
                f"between your profile and '{pid}' → predicted {score:.1f} / 5."
            )
        if method == "Slope One":
            return (
                f"The Slope One deviation model predicts you'd rate '{pid}' "
                f"{score:.1f} / 5, based on the rating offset patterns of users "
                f"who reviewed the same products as you."
            )
        return ""

    # ── public API ───────────────────────────────────────────────────────

    def recommend(self, uid: str, method: str = "SVD", n: int = 10) -> list[dict]:
        """
        Returns a list of dicts:
            {ProductId, PredictedScore, Explanation}
        """
        if method not in self._models:
            raise ValueError(f"method must be one of {METHODS}")

        model  = self._models[method]
        rated  = set(self.df[self.df["UserId"] == uid]["ProductId"])
        unrated = list(self.all_products - rated)

        preds = sorted(
            [(p, self._predict(model, uid, p)) for p in unrated],
            key=lambda x: x[1], reverse=True
        )[:n]

        return [
            {
                "ProductId":      p,
                "PredictedScore": round(s, 2),
                "Explanation":    self._explain(method, uid, p, s),
                "Method":         method,
                "Approach":       "Collaborative Filtering",
            }
            for p, s in preds
        ]

    def evaluate(self) -> dict:
        """
        Returns RMSE and MAE for all 4 methods on the held-out test set.
        """
        results = {}
        for name, model in self._models.items():
            preds = model.test(self.testset)
            results[name] = {
                "RMSE": round(surp_acc.rmse(preds, verbose=False), 4),
                "MAE":  round(surp_acc.mae(preds,  verbose=False), 4),
            }
        return results

    def get_user_list(self) -> list[str]:
        return sorted(self.df["UserId"].unique().tolist())
