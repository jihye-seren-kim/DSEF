# dsef/detectors/random_forest.py
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class RandomForestDetector:
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )
        self.label_col = "label"
        self.feature_cols: List[str] = []

    def _make_labels(self, df: pd.DataFrame) -> np.ndarray:
        y_raw = df[self.label_col].astype(str)
        return (y_raw != "Benign").astype(int).to_numpy()

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        if self.feature_cols:
            return self.feature_cols
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        blacklist = {"label", "set", "split"}
        self.feature_cols = [c for c in cols if c not in blacklist]
        return self.feature_cols

    def fit(self, train_df: pd.DataFrame):
        if self.label_col not in train_df.columns:
            return
        X_cols = self._select_features(train_df)
        if not X_cols:
            return
        X = train_df[X_cols].to_numpy(dtype=float)
        y = self._make_labels(train_df)
        self.model.fit(X, y)

    def evaluate_auc(self, test_df: pd.DataFrame) -> float:
        if self.label_col not in test_df.columns:
            return 0.5
        X_cols = self._select_features(test_df)
        if not X_cols:
            return 0.5
        X = test_df[X_cols].to_numpy(dtype=float)
        y = self._make_labels(test_df)
        try:
            proba = self.model.predict_proba(X)[:, 1]
            return float(roc_auc_score(y, proba))
        except Exception:
            return 0.5
