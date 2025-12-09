# dsef/reporting/metrics_store.py
import json
from typing import Dict


class MetricsStore:
    def __init__(self, run_id: str):
        self.run_id = run_id
        # self.metrics[backbone][axis] = dict(...)
        self.metrics: Dict[str, Dict[str, Dict]] = {}

    def add_axis_metrics(self, backbone: str, axis_name: str, axis_metrics: Dict):
        if backbone not in self.metrics:
            self.metrics[backbone] = {}
        self.metrics[backbone][axis_name] = axis_metrics

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "metrics": self.metrics,
        }

    def save(self, path: str):
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def compute_composite_scores(self, weights_cfg: Dict[str, list], scenario: str = "default") -> Dict[str, float]:
        """
        weights_cfg: config["weights"] (scenario -> [w1, w2, w3, w4])
        scenario: "default" / "mal_phish" / "exfil_heavy" 등
        axis metric dict 안에 "score"라는 key가 있다고 가정
        """
        w = weights_cfg.get(scenario, weights_cfg["default"])
        if len(w) != 4:
            raise ValueError("weights must have length 4 (for 4 axes)")

        composite = {}
        for backbone, axes in self.metrics.items():
            s1 = axes.get("axis1", {}).get("score", 0.0)
            s2 = axes.get("axis2", {}).get("score", 0.0)
            s3 = axes.get("axis3", {}).get("score", 0.0)
            s4 = axes.get("axis4", {}).get("score", 0.0)
            total = w[0] * s1 + w[1] * s2 + w[2] * s3 + w[3] * s4
            composite[backbone] = {
                "S_DSEF": total,
                "axis_scores": [s1, s2, s3, s4],
                "weights": w,
            }
        return composite
