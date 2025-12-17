
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin

MODEL_PATHS = [Path(__file__).parent / "reu-model.joblib",]

def adaptive_difficulty_adjustment(current_difficulty, block_score, network_load):
    """
    Implements the required formula:
    new_difficulty = current * adjustment_factor, clipped to ±10 percent
    """
    adjustment_factor = block_score / (1 + network_load)
    adjustment_factor = np.clip(adjustment_factor, 0.9, 1.1)
    return current_difficulty * adjustment_factor

class CustomConsensusWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, gas_weight=0.4, tx_density_weight=0.3, difficulty_weight=0.3):
        self.gas_weight = gas_weight
        self.tx_density_weight = tx_density_weight
        self.difficulty_weight = difficulty_weight

    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame):
        block_scores = (
            self.gas_weight * X["gas_used"]
            + self.tx_density_weight * X["transaction_count"]
            + self.difficulty_weight / (1 + np.log(X["log_difficulty"] + 1))
        )
        return np.where(block_scores > 0, 1.0 / (block_scores + 1e-9), 1e6)


sys.modules["__main__"] = sys.modules[__name__]
sys.modules.setdefault("model", sys.modules[__name__])


class MLEngine:
    def __init__(self):
        self.model = None
        for p in MODEL_PATHS:
            if p.exists():
                self.model = joblib.load(p)
                break

    def predict_quality(self, sample: dict) -> float:
        # Fallback from observed latency
        def score_from_obs(lat_ms):
            return 1.0 / (1.0 + lat_ms / 300.0)

        difficulty = float(sample.get("difficulty", 1))
        gas_used = float(sample.get("gas_used", 0))
        tx_count = float(sample.get("transaction_count", 0))

        # Compute baseline block_score
        block_score = (
            0.4 * gas_used +
            0.3 * tx_count +
            0.3 / (1 + np.log(difficulty + 1))
        )

        # ------------ NEW PART: Use adaptive difficulty ------------
        network_load = max(0.001, tx_count / (gas_used + 1))  # simple proxy
        new_difficulty = adaptive_difficulty_adjustment(
            difficulty, block_score, network_load
        )
        difficulty_adjustment_score = 1 - (abs(new_difficulty - difficulty) / difficulty)
        # -----------------------------------------------------------

        if self.model is None:
            lat_obs = float(sample.get("latency_ms", 0.0))
            base_score = score_from_obs(lat_obs)
        else:
            input_df = pd.DataFrame([{
                "gas_used": gas_used,
                "transaction_count": tx_count,
                "log_difficulty": np.log(difficulty + 1),
                "block_score": block_score,
            }])

            pred = float(self.model.predict(input_df)[0])
            base_score = 1.0 / (1.0 + 5e6 * max(pred, 1e-12))

        lat_obs = float(sample.get("latency_ms", 0))
        latency_score = score_from_obs(lat_obs)

        # FINAL ML SCORE = hybrid of RF, latency, and teacher's adaptive difficulty
        final_score = 0.4 * base_score + 0.4 * latency_score + 0.2 * difficulty_adjustment_score
        return float(np.clip(final_score, 0.05, 0.99))
