# dsef/generators/ml_backbone.py
import pandas as pd


class MLBackbone:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # TODO: initialize cGAN / VAE / Transformer

    def generate(self, df_real: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
        """
        placeholder: same as tool_backbone.py
        """
        label_col = "label"
        dfs = []
        for label, group in df_real.groupby(label_col):
            sample = group.sample(
                n=min(n_per_class, len(group)),
                replace=len(group) < n_per_class,
                random_state=43,
            )
            dfs.append(sample)
        syn_df = pd.concat(dfs, ignore_index=True)
        syn_df["source"] = "ml"
        return syn_df
