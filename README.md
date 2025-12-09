# DSEF

```
dsef/
  ├── run_experiment.py
  ├── config.yaml
  ├── data/
  │   ├── raw/                    # BCCC–CIC–Bell–DNS–2024
  │   ├── processed/              # preprocessed feature CSV / parquet
  │   └── splits.json             # train/val/test split information
  ├── dsef/
  │   ├── __init__.py
  │   ├── config.py               # YAML loader
  │   ├── data_utils.py           # loading, split, feature extraction
  │   ├── generators/
  │   │   ├── tool_backbone.py
  │   │   ├── ml_backbone.py
  │   │   └── llm_backbone.py
  │   ├── embeddings/
  │   │   ├── autoencoder.py      # AE learning + embedding
  │   │   └── ftd.py              # Fréchet Traffic Distance
  │   ├── validation/
  │   │   ├── axis1_protocol.py
  │   │   ├── axis2_distribution.py
  │   │   ├── axis3_semantic.py
  │   │   └── axis4_utility_diversity.py
  │   ├── detectors/
  │   │   ├── rule_based.py
  │   │   ├── random_forest.py
  │   │   └── flan_t5_wrapper.py
  │   └── reporting/
  │       ├── metrics_store.py    # save JSON
  │       └── plot_all.py
  └── manifests/
      └── run_YYYYMMDD_HHMM.json
```
