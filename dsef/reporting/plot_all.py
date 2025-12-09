# dsef/reporting/plot_all.py
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def load_metrics(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_composite_scores(metrics_dict: Dict, out_path: str):
    # metrics_dict: MetricsStore.compute_composite_scores()
    backbones = list(metrics_dict.keys())
    scores = [metrics_dict[b]["S_DSEF"] for b in backbones]

    plt.figure(figsize=(5, 3))
    plt.bar(backbones, scores)
    plt.ylabel("S_DSEF")
    plt.title("Composite scores per backbone")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.metrics_json)
    # 예: default scenario
    composite = metrics.get("composite_default", {})

    if composite:
        plot_composite_scores(composite, str(out_dir / "composite_scores.png"))
        print(f"[plot_all] saved composite_scores.png to {out_dir}")
    else:
        print("[plot_all] no composite_default found in metrics json")
