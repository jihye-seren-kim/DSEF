# dsef/validation/axis4_utility_diversity.py
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import entropy


def _normalized_entropy(counts: np.ndarray) -> float:
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    h = entropy(p)
    h_max = np.log(len(p)) if len(p) > 0 else 1.0
    return float(h / (h_max + 1e-12))


def _simpson_index(counts: np.ndarray) -> float:
    n = counts.sum()
    if n <= 1:
        return 0.0
    return float(1.0 - np.sum(counts * (counts - 1)) / (n * (n - 1)))


def _unique_ratio(df: pd.DataFrame, cols: List[str]) -> float:
    if len(df) == 0:
        return 0.0
    return float(df[cols].drop_duplicates().shape[0] / len(df))


def compute_utility_diversity_metrics(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    syn_train: pd.DataFrame,
    syn_test: pd.DataFrame,
    detectors: Dict[str, object],
) -> Dict[str, float]:
    """
    Axis 4: Downstream utility (TRTS/TSTR) + diversity proxies.

    detectors: dict[name -> detector]
      detector must implement:
        - fit(train_df)
        - evaluate_auc(test_df) -> float
    """
    metrics: Dict[str, float] = {}

    # ---------- TRTS / TSTR ----------
    for name, det in detectors.items():
        # TRTS: train real, test syn
        det.fit(real_train)
        trts_auc = det.evaluate_auc(syn_test)

        # TSTR: train syn, test real
        det.fit(syn_train)
        tstr_auc = det.evaluate_auc(real_test)

        metrics[f"{name}_trts_auc"] = trts_auc
        metrics[f"{name}_tstr_auc"] = tstr_auc

    # AUC
    auc_vals = [
        v for k, v in metrics.items() if k.endswith("_trts_auc") or k.endswith("_tstr_auc")
    ]
    if auc_vals:
        metrics["utility_auc_mean"] = float(np.mean(auc_vals))
    else:
        metrics["utility_auc_mean"] = 0.0

    # ---------- Diversity ----------
    cat_cols = [c for c in ["label", "qtype", "rcode"] if c in syn_train.columns]
    if cat_cols:
        # class-conditional counts
        for col in cat_cols:
            counts = syn_train[col].value_counts().to_numpy()
            nent = _normalized_entropy(counts)
            simp = _simpson_index(counts)
            metrics[f"div_{col}_norm_entropy"] = nent
            metrics[f"div_{col}_simpson"] = simp

    # global unique-flow ratio (numeric feature subset)
    num_cols = syn_train.select_dtypes(include=[np.number]).columns.tolist()
    blacklist = {"label", "set", "split"}
    num_cols = [c for c in num_cols if c not in blacklist]
    if num_cols:
        u_ratio = _unique_ratio(syn_train, num_cols)
        metrics["unique_flow_ratio"] = u_ratio
    else:
        metrics["unique_flow_ratio"] = 0.0

    # Axis 4 score: utility + diversity 
    s_utility = metrics["utility_auc_mean"] 
    s_div = float(
        np.mean(
            [
                v
                for k, v in metrics.items()
                if "norm_entropy" in k or "simpson" in k or k == "unique_flow_ratio"
            ]
        )
    ) if any("norm_entropy" in k or "simpson" in k for k in metrics) else 0.0

    score = 0.6 * s_utility + 0.4 * s_div
    metrics["score_utility"] = s_utility
    metrics["score_diversity"] = s_div
    metrics["score"] = float(score)

    return metrics
