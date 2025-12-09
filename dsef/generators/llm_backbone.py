# dsef/generators/llm_backbone.py
import pandas as pd


class LLMBackbone:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # TODO: set LLM client / prompt template

    def generate(self, df_real: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
        """
        sampling placeholder.
        LLM qname / payload augment + temporal/statistical features
        """
        label_col = "label"
        dfs = []
        for label, group in df_real.groupby(label_col):
            sample = group.sample(
                n=min(n_per_class, len(group)),
                replace=len(group) < n_per_class,
                random_state=44,
            )
            dfs.append(sample)
        syn_df = pd.concat(dfs, ignore_index=True)
        syn_df["source"] = "llm"
        return syn_df
