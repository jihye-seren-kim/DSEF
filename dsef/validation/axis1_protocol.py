# dsef/validation/axis1_protocol.py
from typing import Dict

import numpy as np
import pandas as pd


def _estimate_violation_rate(df: pd.DataFrame) -> float:
    """
    Very lightweight proxy for DNS/EDNS(0) protocol correctness.
    This is intentionally simple so that it works with the
    BCCC–CIC–Bell–DNS–2024 feature schema without raw packets.

    Heuristics (only applied if the column exists):
      - qname_len > 255  -> violation
      - qname_labels > 63 -> violation (very conservative)
      - ttl_min < 0 or ttl_max < 0 -> violation
    """
    if len(df) == 0:
        return 0.0

    violation = np.zeros(len(df), dtype=bool)

    if "qname_len" in df.columns:
        violation |= df["qname_len"].to_numpy() > 255

    if "qname_labels" in df.columns:
        violation |= df["qname_labels"].to_numpy() > 63

    for col in ["ttl_min", "ttl_max"]:
        if col in df.columns:
            violation |= df[col].to_numpy() < 0

    # NOTE: if more protocol fields are available later
    # (e.g., rcode, qclass, edns_do, edns_cd, ancount) you can
    # add them here following the paper's A1.x checklist.

    rate = float(violation.mean())
    return rate


def compute_protocol_metrics(
    real_flows: pd.DataFrame,
    syn_flows: pd.DataFrame,
) -> Dict[str, float]:
    """
    Axis 1: Protocol correctness.

    Returns
    -------
    metrics : dict
      {
        "violation_rate_real": ...,
        "violation_rate_syn": ...,
        "s_prot": 1 - violation_rate_syn,
        "score": same as s_prot (for composite)
      }
    """
    nu_real = _estimate_violation_rate(real_flows)
    nu_syn = _estimate_violation_rate(syn_flows)

    s_prot = max(0.0, 1.0 - nu_syn)

    return {
        "violation_rate_real": nu_real,
        "violation_rate_syn": nu_syn,
        "s_prot": s_prot,
        "score": s_prot,
    }
