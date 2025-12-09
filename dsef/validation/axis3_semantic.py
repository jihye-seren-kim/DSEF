# dsef/validation/axis3_semantic.py
from typing import Dict, Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from dsef.embeddings.ftd import compute_ftd
from dsef.embeddings.autoencoder import AutoEncoderTrainer


def _entropy_gap(real: pd.Series, syn: pd.Series) -> float:
    r = real.to_numpy(dtype=float)
    s = syn.to_numpy(dtype=float)
    r = r[~np.isnan(r)]
    s = s[~np.isnan(s)]
    if len(r) == 0 or len(s) == 0:
        return 0.0
    return float(abs(r.mean() - s.mean()))


def compute_semantic_metrics(
    real_flows: pd.DataFrame,
    syn_flows: pd.DataFrame,
    ae,
    feature_cols: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Axis 3: Semantic & behavioral realism.

    Parameters
    ----------
    ae : torch.nn.Module
        Trained AE.
    feature_cols : list[str]
        AE input cols; if None, try to infer from numeric cols.
    """
    if feature_cols is None:
        # fallback: simple numeric subset
        feat_all = real_flows.select_dtypes(include=[np.number]).columns.tolist()
        blacklist = {"label", "set", "split", "source"}
        feature_cols = [c for c in feat_all if c not in blacklist]

    # --- FTD ---
    ftd_val = compute_ftd(ae, real_flows, syn_flows, feature_cols)

    # --- QNAME entropy deviation (if available) ---
    ent_col_candidates = ["qname_entropy", "qname_ent", "qname_entrop"]
    ent_col = next((c for c in ent_col_candidates if c in real_flows.columns), None)
    if ent_col is not None:
        ent_gap = _entropy_gap(real_flows[ent_col], syn_flows[ent_col])
    else:
        ent_gap = 0.0

    # --- Embedding-based clustering (very simple) ---
    Z_r = AutoEncoderTrainer.encode(ae, real_flows, feature_cols)
    Z_s = AutoEncoderTrainer.encode(ae, syn_flows, feature_cols)

    # concat embeddings, cluster, then ARI between "real/syn" labels and cluster labels
    Z = np.vstack([Z_r, Z_s])
    labels_true = np.array([0] * len(Z_r) + [1] * len(Z_s))

    n_clusters = 3
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    cluster_labels = km.fit_predict(Z)

    ari = adjusted_rand_score(labels_true, cluster_labels)

    # centroid cosine similarity between real & syn centroids in latent space
    c_r = Z_r.mean(axis=0)
    c_s = Z_s.mean(axis=0)
    dot = float(np.dot(c_r, c_s))
    denom = float(np.linalg.norm(c_r) * np.linalg.norm(c_s) + 1e-8)
    centroid_cos = dot / denom

    # 점수 매핑: FTD는 작을수록, ARI/코사인은 클수록 좋음
    def _score_from_ftd(v: float, ref: float = 5.0) -> float:
        return float(max(0.0, min(1.0, 1.0 - v / ref)))

    def _clamp01(v: float) -> float:
        return float(max(0.0, min(1.0, v)))

    s_ftd = _score_from_ftd(ftd_val)
    s_ari = _clamp01((ari + 1.0) / 2.0)  # ARI in [-1,1]
    s_centroid = _clamp01((centroid_cos + 1.0) / 2.0)

    score = 0.4 * s_ftd + 0.3 * s_ari + 0.3 * s_centroid

    return {
        "ftd": ftd_val,
        "entropy_gap": ent_gap,
        "ari_real_vs_syn": ari,
        "centroid_cosine": centroid_cos,
        "s_ftd": s_ftd,
        "s_ari": s_ari,
        "s_centroid": s_centroid,
        "score": score,
        "feature_cols": list(feature_cols),
    }
