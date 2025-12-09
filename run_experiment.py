#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime

import numpy as np

from dsef.config import load_config
from dsef.data_utils import load_or_create_splits, load_dataset
from dsef.generators.tool_backbone import ToolBackbone
from dsef.generators.ml_backbone import MLBackbone
from dsef.generators.llm_backbone import LLMBackbone
from dsef.embeddings.autoencoder import AutoEncoderTrainer
from dsef.validation.axis1_protocol import evaluate_protocol_axis
from dsef.validation.axis2_distribution import evaluate_distribution_axis
from dsef.validation.axis3_semantic import evaluate_semantic_axis
from dsef.validation.axis4_utility_diversity import evaluate_utility_diversity_axis
from dsef.reporting.metrics_store import MetricsStore


def parse_args():
    parser = argparse.ArgumentParser(description="Run DSEF experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier (otherwise timestamp).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 1. Run ID + manifest path
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs("manifests", exist_ok=True)
    manifest_path = os.path.join("manifests", f"manifest_{run_id}.json")

    # 2. Data loading + splits
    print("[*] Loading dataset and splits...")
    splits = load_or_create_splits(cfg, save_path="data/splits.json")
    df_train, df_val, df_test = load_dataset(cfg, splits)

    # 3. AutoEncoder (Axis 3)
    print("[*] Training AutoEncoder on real train data...")
    ae_trainer = AutoEncoderTrainer(cfg["embeddings"])
    ae_model, ae_meta = ae_trainer.fit(df_train)

    # 4. Init generators
    print("[*] Initializing generators...")
    tool_gen = ToolBackbone(cfg["generators"]["tool"])
    ml_gen = MLBackbone(cfg["generators"]["ml"])
    llm_gen = LLMBackbone(cfg["generators"]["llm"])

    backbones = {
        "tool": tool_gen,
        "ml": ml_gen,
        "llm": llm_gen,
    }

    metrics_store = MetricsStore(run_id=run_id)

    # 5. Per-backbone evaluation
    for name, gen in backbones.items():
        print(f"\n[***] Evaluating backbone: {name.upper()}")

        # 5.1 Generate synthetic flows in unified schema
        syn_df = gen.generate(df_train, n_per_class=cfg["n_per_class"])

        # 5.2 Axis 1: Protocol correctness
        print("  [A1] Protocol correctness...")
        axis1_res = evaluate_protocol_axis(cfg, syn_df)
        metrics_store.add_axis_metrics(name, "axis1", axis1_res)

        # 5.3 Axis 2: Distributional realism
        print("  [A2] Distributional realism...")
        axis2_res = evaluate_distribution_axis(cfg, df_train, syn_df)
        metrics_store.add_axis_metrics(name, "axis2", axis2_res)

        # 5.4 Axis 3: Semantic & behavioral realism
        print("  [A3] Semantic & behavioral realism...")
        axis3_res = evaluate_semantic_axis(cfg, df_train, syn_df, ae_model)
        metrics_store.add_axis_metrics(name, "axis3", axis3_res)

        # 5.5 Axis 4: Utility & diversity
        print("  [A4] Utility & diversity (TRTS/TSTR)...")
        axis4_res = evaluate_utility_diversity_axis(
            cfg, df_train, df_test, syn_df
        )
        metrics_store.add_axis_metrics(name, "axis4", axis4_res)

    # 6. Composite scores
    print("\n[*] Computing composite scores...")
    composite = metrics_store.compute_composite_scores(cfg["weights"])
    metrics_store.save("results_metrics.json")

    # 7. Save manifest
    manifest = {
        "run_id": run_id,
        "config": cfg,
        "splits": splits,
        "ae_meta": ae_meta,
        "composite": composite,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": cfg.get("seed", 42),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] Done. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
