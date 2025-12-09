# dsef/validation/axis2_distribution.py
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def _select_numeric_features(df: pd.DataFrame) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # meta / non-feature columns 제외
    blacklist = {"label", "set", "split", "source"}
    return [c for c in cols if c not in blacklist]


def _hist_jsd(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if lo == hi:
        return 0.0

    hist_x, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hist_y, _ = np.histogram(y, bins=bins, range=(lo, hi), density=True)

    p = hist_x + 1e-12
    q = hist_y + 1e-12
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def _mean_mmd_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Very simple 'linear-kernel MMD' between feature vectors.
    Enough as a placeholder, cheap to compute.
    """
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    return float(np.linalg.norm(mu_x - mu_y))


def compute_distribution_metrics(
    real_flows: pd.DataFrame,
    syn_flows: pd.DataFrame,
) -> Dict[str, float]:
    """
    Axis 2: Distributional realism.
    - per-feature JSD, Wasserstein
    - simple multivariate MMD-style distance
    """
    feat_cols = _select_numeric_features(real_flows)
    if not feat_cols:
        return {"score": 0.0}

    jsd_list = []
    wd_list = []

    for col in feat_cols:
        x = real_flows[col].to_numpy(dtype=float)
        y = syn_flows[col].to_numpy(dtype=float)

        jsd_val = _hist_jsd(x, y)
        wd_val = wasserstein_distance(x, y)

        jsd_list.append(jsd_val)
        wd_list.append(wd_val)

    X = real_flows[feat_cols].to_numpy(dtype=float)
    Y = syn_flows[feat_cols].to_numpy(dtype=float)
    mmd_lin = _mean_mmd_linear(X, Y)

    jsd_mean = float(np.mean(jsd_list))
    wd_mean = float(np.mean(wd_list))

    # sample
    def _to_score(d: float, ref: float = 0.1) -> float:
        return float(max(0.0, min(1.0, 1.0 - d / ref)))

    score_jsd = _to_score(jsd_mean, ref=0.1)
    score_mmd = _to_score(mmd_lin, ref=1.0)
    score = 0.5 * score_jsd + 0.5 * score_mmd

    return {
        "features_used": feat_cols,
        "jsd_mean": jsd_mean,
        "wd_mean": wd_mean,
        "mmd_linear": mmd_lin,
        "score_jsd": score_jsd,
        "score_mmd": score_mmd,
        "score": score,
    }
