# ml_engine/model.py
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin

# Try these paths in order
MODEL_PATHS = [
    Path(__file__).parent / "reu-model3.joblib",
    Path(__file__).parent / "reu-model2.joblib",
    Path(__file__).parent / "reu-model.joblib",
]

class CustomConsensusWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        gas_weight=0.4,
        tx_density_weight=0.3,
        difficulty_weight=0.3,
        max_adjustment=0.1,
    ):
        self.gas_weight = gas_weight
        self.tx_density_weight = tx_density_weight
        self.difficulty_weight = difficulty_weight
        self.max_adjustment = max_adjustment

    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame):
        block_scores = (
            self.gas_weight * X["gas_used"]
            + self.tx_density_weight * X["transaction_count"]
            + self.difficulty_weight / (1 + np.log(X["log_difficulty"] + 1))
        )
        # Inverse block score as "latency proxy"
        return np.where(block_scores > 0, 1.0 / (block_scores + 1e-9), 1e6)

# Hack to fix models saved as __main__.CustomConsensusWrapper
sys.modules["__main__"] = sys.modules[__name__]
# Also fix old saves like model.CustomConsensusWrapper
sys.modules.setdefault("model", sys.modules[__name__])

class MLEngine:
    def __init__(self):
        self.model = None
        for p in MODEL_PATHS:
            if p.exists():
                self.model = joblib.load(p)
                break

    def predict_quality(self, sample: dict) -> float:
        # Fallback based on observed latency only
        def score_from_obs(lat_ms: float) -> float:
            # 300 ms half-life
            return 1.0 / (1.0 + lat_ms / 300.0)

        if self.model is None:
            lat_obs = float(sample.get("latency_ms", 0.0))
            return float(np.clip(score_from_obs(lat_obs), 0.05, 0.99))

        # Build features
        input_df = pd.DataFrame(
            [
                {
                    "gas_used": sample.get("gas_used", 0),
                    "transaction_count": sample.get("transaction_count", 0),
                    "log_difficulty": np.log(sample.get("difficulty", 1) + 1),
                    "block_score": (
                        0.4 * sample.get("gas_used", 0)
                        + 0.3 * sample.get("transaction_count", 0)
                        + 0.3 / (1 + np.log(sample.get("difficulty", 1) + 1))
                    ),
                }
            ]
        )

        # Model output is tiny; rescale aggressively to spread [0.1, 0.95]
        pred = float(self.model.predict(input_df)[0])  # ~ 1 / block_score
        score_pred = 1.0 / (1.0 + 5e6 * max(pred, 1e-12))  # tune 5e6 if needed

        lat_obs = float(sample.get("latency_ms", 0.0))
        score_obs = score_from_obs(lat_obs)

        score = 0.5 * score_pred + 0.5 * score_obs
        return float(np.clip(score, 0.05, 0.99))

# import sys, joblib, pandas as pd, numpy as np
# from pathlib import Path
# from sklearn.base import BaseEstimator, RegressorMixin

# MODEL_PATH = Path(__file__).parent / "reu-model3.joblib"

# class CustomConsensusWrapper(BaseEstimator, RegressorMixin):
#     def __init__(self, gas_weight=0.4, tx_density_weight=0.3,
#                  difficulty_weight=0.3, max_adjustment=0.1):
#         self.gas_weight = gas_weight
#         self.tx_density_weight = tx_density_weight
#         self.difficulty_weight = difficulty_weight
#         self.max_adjustment = max_adjustment
#     def fit(self, X, y=None): return self
#     def predict(self, X):
#         block_scores = (self.gas_weight * X["gas_used"] +
#                         self.tx_density_weight * X["transaction_count"] +
#                         self.difficulty_weight/(1+np.log(X["log_difficulty"]+1)))
#         return np.where(block_scores > 0, 1/(block_scores+1e-9), 1e6)

# # 👇 Alias "__main__" to this module so pickle can find the class
# sys.modules["__main__"] = sys.modules[__name__]

# class MLEngine:
#     def __init__(self):
#         self.model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

#     def predict_quality(self, sample: dict) -> float:
#         if not self.model: return 1.0
#         input_df = pd.DataFrame([{
#             "gas_used": sample.get("gas_used", 0),
#             "transaction_count": sample.get("transaction_count", 0),
#             "log_difficulty": np.log(sample.get("difficulty", 1) + 1),
#             "block_score": (
#                 0.4 * sample.get("gas_used", 0)
#                 + 0.3 * sample.get("transaction_count", 0)
#                 + 0.3/(1+np.log(sample.get("difficulty", 1) + 1))
#             )
#         }])
#         pred_latency = float(self.model.predict(input_df)[0])
#         return 1.0 / (1.0 + pred_latency)

# # import joblib
# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # from sklearn.base import BaseEstimator, RegressorMixin
# # from sklearn.ensemble import RandomForestRegressor

# # MODEL_PATH = Path(__file__).parent / "reu-model.joblib"

# # # Copy-paste exactly the same class you trained
# # rf_model = RandomForestRegressor(n_estimators=50, random_state=0)  # dummy filler to satisfy fit
# # class CustomConsensusWrapper(BaseEstimator, RegressorMixin):
# #     def __init__(self, gas_weight=0.4, tx_density_weight=0.3,
# #                  difficulty_weight=0.3, max_adjustment=0.1):
# #         self.gas_weight = gas_weight
# #         self.tx_density_weight = tx_density_weight
# #         self.difficulty_weight = difficulty_weight
# #         self.max_adjustment = max_adjustment

# #     def fit(self, X, y=None):
# #         difficulty_predictions = rf_model.predict(X)
# #         self.difficulty_weight = np.clip(np.mean(difficulty_predictions) / 1e5, 0.1, 1.0)
# #         return self

# #     def predict(self, X):
# #         block_scores = (self.gas_weight * X['gas_used'] +
# #                         self.tx_density_weight * X['transaction_count'] +
# #                         self.difficulty_weight / (1 + np.log(X['log_difficulty'] + 1)))
# #         latency_predictions = np.where(block_scores > 0,
# #                                        1/(block_scores+1), np.inf)
# #         latency_predictions = np.nan_to_num(latency_predictions, nan=1e6,
# #                                             posinf=1e6, neginf=1e6)
# #         return latency_predictions

# #     def score(self, X, y):
# #         predictions = self.predict(X)
# #         from sklearn.metrics import mean_absolute_error
# #         return -mean_absolute_error(y, predictions)

# # # Wrapper engine
# # class MLEngine:
# #     def __init__(self):
# #         if MODEL_PATH.exists():
# #             self.model = joblib.load(MODEL_PATH)   # ✅ will now match
# #             self.ready = True
# #         else:
# #             self.model = None
# #             self.ready = False

# #     def predict_quality(self, sample: dict) -> float:
# #         if not self.ready: 
# #             return 1.0

# #         # Must match training features
# #         input_df = pd.DataFrame([{
# #             "gas_used": sample.get("gas_used", 0),
# #             "transaction_count": sample.get("transaction_count", 0),
# #             "log_difficulty": np.log(sample.get("difficulty", 1) + 1),
# #             "block_score": (0.4 * sample.get("gas_used", 0) +
# #                             0.3 * sample.get("transaction_count", 0) +
# #                             0.3 / (1 + np.log(sample.get("difficulty", 1) + 1)))
# #         }])

# #         pred_latency = float(self.model.predict(input_df)[0])
# #         return 1.0 / (1.0 + pred_latency)
    
# # # import joblib
# # # from pathlib import Path
# # # from .features import build_features

# # # MODEL_PATH = Path(__file__).with_name("reu-model.joblib")

# # # class MLEngine:
# # #     def __init__(self):
# # #         if MODEL_PATH.exists():
# # #             self.model = joblib.load(MODEL_PATH)
# # #             self.ready = True
# # #         else:
# # #             self.model = None
# # #             self.ready = False

# # #     def predict_quality(self, sample: dict) -> float:
# # #         if self.ready and self.model is not None:
# # #             X = build_features(sample)
# # #             pred_latency = float(self.model.predict(X)[0])
# # #             score = 1.0 / (1.0 + pred_latency / 1000.0)
# # #             return max(0.0, min(1.0, score))
# # #         latency = float(sample.get("latency_ms", 0))
# # #         return 1.0 / (1.0 + latency / 1000.0)