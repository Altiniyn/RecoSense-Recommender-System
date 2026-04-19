"""
evaluation/evaluator.py
────────────────────────
Comprehensive evaluation of all recommendation approaches.

Metrics implemented (≥ 4 as required):
  1. RMSE              – root-mean-squared error on held-out ratings (CF)
  2. MAE               – mean-absolute error on held-out ratings      (CF)
  3. Precision @ N     – fraction of top-N recs that are relevant
  4. Recall @ N        – fraction of relevant items found in top-N
  5. Coverage %        – fraction of catalogue surfaced across users
  6. Novelty           – mean self-information of recommended items
                         (higher = more niche / surprising)

BUG FIXED:
  • Classic Python lambda-closure bug in Evaluator.run():
    `lambda u: cf_engine.recommend(u, method, n)` inside a loop captures
    `method` by reference, so every lambda ends up calling the *last*
    value of `method` ("Slope One").
    Fix: use a default-argument to snapshot the value at creation time:
    `lambda u, _m=method: cf_engine.recommend(u, _m, n)`
    Same fix applied to all CB and KB lambdas.

  • to_dataframe(): RMSE_MAE row used hard-coded key lookup that would
    KeyError if either key was absent; now uses .get() with a "-" fallback.
"""

import numpy as np
import pandas as pd
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────

def _temporal_split(df: pd.DataFrame, test_ratio: float = 0.2):
    if "Time" in df.columns and df["Time"].nunique() > 1:
        df_s = df.sort_values("Time")
    else:
        df_s = df.sample(frac=1, random_state=42)
    cut = int((1 - test_ratio) * len(df_s))
    return df_s.iloc[:cut].copy(), df_s.iloc[cut:].copy()


def _ranking_metrics(
    recs_fn:      Callable[[str], list[dict]],
    df_test:      pd.DataFrame,
    all_products: set,
    df_full:      pd.DataFrame,
    n:            int = 10,
    max_users:    int = 200,
    label:        str = "",
) -> dict:
    """Precision@N, Recall@N, Coverage%, Novelty for one recommender."""
    popularity = df_full["ProductId"].value_counts()
    prec_list, rec_list, cov_set, nov_list = [], [], set(), []

    for uid in df_test["UserId"].unique()[:max_users]:
        relevant = set(
            df_test[(df_test["UserId"] == uid) & (df_test["Score"] >= 4)]["ProductId"]
        )
        if not relevant:
            continue
        try:
            raw = recs_fn(uid) or []
        except Exception:
            continue

        rec_pids = [r["ProductId"] for r in raw][:n]
        if not rec_pids:
            continue

        hits = len(set(rec_pids) & relevant)
        prec_list.append(hits / n)
        rec_list.append(hits / len(relevant))
        cov_set.update(rec_pids)

        for pid in rec_pids:
            pop = max(popularity.get(pid, 1), 1)
            nov_list.append(-np.log2(pop / len(df_full)))

    return {
        f"Precision@{n}": round(np.mean(prec_list), 4) if prec_list else 0.0,
        f"Recall@{n}":    round(np.mean(rec_list),  4) if rec_list  else 0.0,
        "Coverage(%)":    round(100 * len(cov_set) / max(len(all_products), 1), 2),
        "Novelty":        round(np.mean(nov_list),   4) if nov_list  else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:

    def __init__(self, df: pd.DataFrame, n: int = 10):
        self.df_full   = df
        self.n         = n
        self.all_prods = set(df["ProductId"].unique())
        self.df_train, self.df_test = _temporal_split(df)

    # ── main entry point ─────────────────────────────────────────────────

    def run(self, cf_engine, cb_engine, kb_engine) -> dict:
        """
        Returns a nested dict:
        {
          "CF": {
            "RMSE_MAE": {...},
            "SVD": {...},
            "User-Based KNN": {...},
            ...
          },
          "Content-Based": { ... },
          "Knowledge-Based": { ... },
        }
        """
        n   = self.n
        cfg = {
            "df_test":      self.df_test,
            "all_products": self.all_prods,
            "df_full":      self.df_full,
            "n":            n,
        }

        # ── 1-2: RMSE / MAE (CF only) ────────────────────────────────────
        rmse_mae = cf_engine.evaluate()

        # ── 3-6: Ranking metrics ─────────────────────────────────────────
        # BUG FIX: use default-arg `_m=method` to snapshot loop variable.
        # Without this every lambda captures the same final value of `method`.

        cf_methods = ["SVD", "User-Based KNN", "Item-Based KNN", "Slope One"]
        cf_results = {"RMSE_MAE": rmse_mae}
        for method in cf_methods:
            cf_results[method] = _ranking_metrics(
                lambda u, _m=method: cf_engine.recommend(u, _m, n),   # ← FIX
                **cfg, label=method[:3],
            )

        cb_methods = ["TF-IDF", "Weighted-TF-IDF"]
        cb_results = {}
        for method in cb_methods:
            cb_results[method] = _ranking_metrics(
                lambda u, _m=method: cb_engine.recommend(u, _m, n),   # ← FIX
                **cfg, label=method[:3],
            )

        kb_rankings = ["score", "helpfulness", "popularity"]
        kb_labels   = {"score": "Score-ranked",
                       "helpfulness": "Helpfulness-ranked",
                       "popularity":  "Popularity-ranked"}
        kb_results  = {}
        for ranking in kb_rankings:
            kb_results[kb_labels[ranking]] = _ranking_metrics(
                lambda u, _r=ranking: kb_engine.recommend(u, ranking=_r, n=n),  # ← FIX
                **cfg, label=f"KB-{ranking[:1].upper()}",
            )

        return {
            "CF":              cf_results,
            "Content-Based":   cb_results,
            "Knowledge-Based": kb_results,
        }

    # ── convenience: flat table for display ──────────────────────────────

    @staticmethod
    def to_dataframe(report: dict) -> pd.DataFrame:
        rows = []
        for approach, methods in report.items():
            for method, vals in methods.items():
                if method == "RMSE_MAE":
                    for m2, v2 in vals.items():
                        rows.append({
                            "Approach":    approach,
                            "Method":      m2,
                            "RMSE":        v2.get("RMSE", "-"),   # ← FIX: .get() not []
                            "MAE":         v2.get("MAE",  "-"),
                            "Precision@N": "-",
                            "Recall@N":    "-",
                            "Coverage(%)": "-",
                            "Novelty":     "-",
                        })
                else:
                    pk = next((k for k in vals if "Precision" in k), None)
                    rk = next((k for k in vals if "Recall"    in k), None)
                    rows.append({
                        "Approach":    approach,
                        "Method":      method,
                        "RMSE":        vals.get("RMSE", "-"),
                        "MAE":         vals.get("MAE",  "-"),
                        "Precision@N": vals.get(pk, "-") if pk else "-",
                        "Recall@N":    vals.get(rk, "-") if rk else "-",
                        "Coverage(%)": vals.get("Coverage(%)", "-"),
                        "Novelty":     vals.get("Novelty", "-"),
                    })
        return pd.DataFrame(rows)

    # ── analysis text ─────────────────────────────────────────────────────

    @staticmethod
    def analysis_text() -> str:
        return """
**Key Findings**

| Dimension | Winner | Reason |
|---|---|---|
| Rating accuracy (RMSE/MAE) | **SVD** | Latent-factor decomposition captures nuanced user–item interactions that neighbourhood methods miss on sparse matrices |
| Top-N Precision | **Item-Based KNN** | Item neighbourhoods are more stable than user ones; item similarity generalises better |
| Coverage & Novelty | **Content-Based** | Text similarity surfaces niche products with few ratings; no popularity bias |
| Transparency | **Knowledge-Based** | Every recommendation can be traced directly to a user-specified constraint |
| Cold-start (new user) | **Knowledge-Based** | No rating history needed — only user constraints |
| Cold-start (new item) | **Content-Based** | Can recommend items with no ratings using review text |

**When to use each approach**

- **New user / no history** → Knowledge-Based  
- **Rich rating history** → SVD (CF)  
- **New products / sparse ratings** → Content-Based (TF-IDF)  
- **Explainability required** → Knowledge-Based > Content-Based  
- **Discovery / serendipity** → Weighted-TF-IDF  
- **Production accuracy-first** → SVD  
"""