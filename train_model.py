import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib
from ml_engine.features import build_features

def main():
    # If you have your CSV, put it next to this script and rename here
    csv_path = Path("updated_dataset3.csv")
    rows = []
    targets = []

    if csv_path.exists():
        df = pd.read_csv(csv_path).fillna(0)
        for _, r in df.iterrows():
            sample = {
                "gas_used": float(r.get("gas_used", 0)),
                "gas_limit": float(r.get("gas_limit", 30_000_000)),
                "transaction_count": float(r.get("transaction_count", 1)),
                "difficulty": float(r.get("difficulty", 1e12)),
                "total_difficulty": float(r.get("total_difficulty", 1e12)),
                "latency_ms": float(r.get("latency_ms", 1000.0 * (r.get("gas_used", 1) / (r.get("transaction_count", 1) + 1)))),
            }
            X = build_features(sample)
            rows.append(X.iloc[0])
            targets.append(sample["latency_ms"])
    else:
        # Synthetic small dataset
        rng = np.random.default_rng(42)
        for _ in range(1000):
            gas_used = float(rng.integers(1_000_000, 15_000_000))
            tx_count = float(rng.integers(1, 200))
            diff = float(rng.uniform(1e10, 1e13))
            sample = {
                "gas_used": gas_used,
                "gas_limit": 30_000_000.0,
                "transaction_count": tx_count,
                "difficulty": diff,
                "total_difficulty": diff * 2,
                "latency_ms": 1000.0 * (gas_used / (tx_count + 5.0)) / 1e6 + rng.normal(0, 10),
            }
            X = build_features(sample)
            rows.append(X.iloc[0])
            targets.append(sample["latency_ms"])

    X_df = pd.DataFrame(rows).fillna(0)
    y = pd.Series(targets).clip(lower=0, upper=20000)

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_df, y)

    out_path = Path("ml_engine") / "model.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved {out_path.resolve()} with {len(X_df)} samples")

if __name__ == "__main__":
    main()