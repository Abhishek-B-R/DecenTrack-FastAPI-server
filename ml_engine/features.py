import numpy as np
import pandas as pd

NUMERIC_DEFAULTS = {
    "gas_used": 0,
    "gas_limit": 1,
    "transaction_count": 0,
    "difficulty": 1.0,
    "total_difficulty": 1.0,
    "latency_ms": 0.0,
}

FEATURE_COLS = [
    "gas_used",
    "transaction_count",
    "log_difficulty",
    "block_score",
    "gas_transaction_ratio",
    "log_total_difficulty",
    "difficulty_gas_interaction",
    "latency_ms",
]

def build_features(sample: dict) -> pd.DataFrame:
    s = {k: sample.get(k, v) for k, v in NUMERIC_DEFAULTS.items()}
    s["log_difficulty"] = np.log(s["difficulty"] + 1)
    s["transaction_density"] = s["transaction_count"] / (s["gas_used"] + 1)
    s["block_score"] = (
        0.4 * s["gas_used"]
        + 0.3 * s["transaction_count"]
        + 0.3 / (1 + np.log(s["difficulty"] + 1))
    )
    s["gas_transaction_ratio"] = s["gas_used"] / (s["transaction_count"] + 1)
    s["log_total_difficulty"] = np.log(s["total_difficulty"] + 1)
    s["difficulty_gas_interaction"] = s["difficulty"] * s["gas_used"]
    row = {c: s.get(c, 0) for c in FEATURE_COLS}
    return pd.DataFrame([row])