# dsef/detectors/rule_based.py
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class RuleBasedDetector:
    """
    Very simple entropy-based rule detector.

    - Treats non-Benign as malicious.
    - Learns qname_entropy threshold from real benign traffic.
    """
    def __init__(self):
        self.threshold: Optional[float] = None
        self.label_col = "label"

    def _make_labels(self, df: pd.DataFrame) -> np.ndarray:
        y_raw = df[self.label_col].astype(str)
        # Benign vs others
        return (y_raw != "Benign").astype(int).to_numpy()

    def fit(self, train_df: pd.DataFrame):
        if "qname_entropy" not in train_df.columns:
            # fallback: no training
            self.threshold = None
            return

        benign = train_df[train_df[self.label_col].astype(str) == "Benign"]
        if len(benign) == 0:
            benign = train_df

        ent = benign["qname_entropy"].to_numpy(dtype=float)
        self.threshold = float(np.percentile(ent, 95))

    def _score(self, df: pd.DataFrame) -> np.ndarray:
        if "qname_entropy" not in df.columns or self.threshold is None:
            # no info → constant score
            return np.zeros(len(df), dtype=float) + 0.5
        ent = df["qname_entropy"].to_numpy(dtype=float)
        # higher entropy -> more likely malicious
        return (ent - self.threshold) / (np.abs(self.threshold) + 1e-6) + 0.5

    def evaluate_auc(self, test_df: pd.DataFrame) -> float:
        if self.label_col not in test_df.columns:
            return 0.5
        y = self._make_labels(test_df)
        scores = self._score(test_df)
        try:
            return float(roc_auc_score(y, scores))
        except ValueError:
            # only one class in y
            return 0.5
