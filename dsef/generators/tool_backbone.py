# dsef/generators/tool_backbone.py
import pandas as pd


class ToolBackbone:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate(self, df_real: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
        """
        현재는 그냥 real 데이터를 class별로 샘플링하는 placeholder.
        나중에 TRex/MoonGen replay 결과를 flow-level로 aggregate해서 반환하도록 교체.
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
