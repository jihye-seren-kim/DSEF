# dsef/generators/tool_backbone.py
import pandas as pd


class ToolBackbone:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate(self, df_real: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
        """
        placeholder: real data -> class sampling
        TRex/MoonGen replay result - flow-level aggregate later
        """
        label_col = "label"
        dfs = []
        for label, group in df_real.groupby(label_col):
            sample = group.sample(
                n=min(n_per_class, len(group)),
                replace=len(group) < n_per_class,
                random_state=42,
            )
            dfs.append(sample)
        syn_df = pd.concat(dfs, ignore_index=True)
        syn_df["source"] = "tool"  # generator tag
        return syn_df
