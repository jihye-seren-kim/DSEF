# dsef/embeddings/ftd.py (Frechet Traffic Distance)
import numpy as np
from typing import Sequence, Optional

from .autoencoder import AutoEncoderTrainer


def _frechet_distance(mu_r, sigma_r, mu_s, sigma_s, eps: float = 1e-6) -> float:
    """
    Standard Fréchet distance between two Gaussian embeddings.
    """
    # Numerical stability
    sigma_r = sigma_r + np.eye(sigma_r.shape[0]) * eps
    sigma_s = sigma_s + np.eye(sigma_s.shape[0]) * eps

    # sqrtm via eigen-decomposition
    vals, vecs = np.linalg.eigh(sigma_r @ sigma_s)
    vals = np.clip(vals, 0.0, None)
    sqrt_part = vecs @ np.diag(np.sqrt(vals)) @ vecs.T

    diff = mu_r - mu_s
    dist = diff.dot(diff) + np.trace(sigma_r + sigma_s - 2.0 * sqrt_part)
    return float(dist)


def compute_ftd(
    ae_model,
    df_real,
    df_syn,
    feature_cols: Sequence[str],
) -> float:
    """
    Compute Frechet Traffic Distance between real & synthetic flows
    in the AE latent space.

    Parameters
    ----------
    ae_model : torch.nn.Module
        Trained autoencoder (SimpleAE).
    df_real, df_syn : pd.DataFrame
        Real / synthetic flow tables.
    feature_cols : list[str]
        Columns used as AE input.

    Returns
    -------
    ftd : float
    """
    # Encode
    Z_r = AutoEncoderTrainer.encode(ae_model, df_real, feature_cols)
    Z_s = AutoEncoderTrainer.encode(ae_model, df_syn, feature_cols)

    mu_r = Z_r.mean(axis=0)
    mu_s = Z_s.mean(axis=0)
    sigma_r = np.cov(Z_r, rowvar=False)
    sigma_s = np.cov(Z_s, rowvar=False)

    return _frechet_distance(mu_r, sigma_r, mu_s, sigma_s)
