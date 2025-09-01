from pathlib import Path
import joblib
from .features import build_features

MODEL_PATH = Path(__file__).with_name("model.joblib")

class MLEngine:
    def __init__(self):
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            self.ready = True
        else:
            # Fallback lightweight heuristic if model not trained yet
            self.model = None
            self.ready = False

    def predict_quality(self, sample: dict) -> float:
        # If trained model exists, predict latency_ms and convert to [0,1] score
        if self.ready and self.model is not None:
            X = build_features(sample)
            pred_latency = float(self.model.predict(X)[0])
            pred_latency = max(pred_latency, 0.0)
            score = 1.0 / (1.0 + pred_latency / 1000.0)
            return max(0.0, min(1.0, score))
        # Heuristic fallback: lower latency => higher score
        latency = float(sample.get("latency_ms", 0))
        return max(0.0, min(1.0, 1.0 / (1.0 + latency / 1000.0)))