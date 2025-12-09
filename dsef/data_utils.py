# dsef/data_utils.py
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _load_processed(cfg: dict) -> pd.DataFrame:
    proc_dir = cfg["data"]["processed_dir"]
    main_file = cfg["data"]["main_file"]
    path = os.path.join(proc_dir, main_file)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    return df


def load_or_create_splits(cfg: dict, save_path: str) -> Dict[str, Dict[str, list]]:
    """
    returns:
      {
        "train": {"idx": [...]},
        "val":   {"idx": [...]},
        "test":  {"idx": [...]}
      }
    """
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            splits = json.load(f)
        return splits

    df = _load_processed(cfg)
    time_col = cfg["data"].get("time_column", None)
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]
    seed = cfg.get("seed", 42)

    if time_col and time_col in df.columns:
        df_sorted = df.sort_values(time_col)
    else:
        df_sorted = df.copy()

    idx_all = df_sorted.index.to_numpy()

    # train+val vs test
    train_val_idx, test_idx = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        shuffle=False, 
    )

    # train vs val
    val_ratio_in_tv = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_in_tv,
        random_state=seed,
        shuffle=False,
    )

    splits = {
        "train": {"idx": train_idx.tolist()},
        "val": {"idx": val_idx.tolist()},
        "test": {"idx": test_idx.tolist()},
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


def load_dataset(cfg: dict, splits: Dict[str, Dict[str, list]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _load_processed(cfg)
    idx_train = splits["train"]["idx"]
    idx_val = splits["val"]["idx"]
    idx_test = splits["test"]["idx"]

    df_train = df.loc[idx_train].reset_index(drop=True)
    df_val = df.loc[idx_val].reset_index(drop=True)
    df_test = df.loc[idx_test].reset_index(drop=True)

    return df_train, df_val, df_test
