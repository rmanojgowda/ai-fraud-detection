import joblib
import numpy as np
import pandas as pd
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

# ── Load model artifacts once at startup ─────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))

with open(os.path.join(BASE_DIR, "models", "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)

with open(os.path.join(BASE_DIR, "models", "threshold.json")) as f:
    threshold_data = json.load(f)
    THRESHOLD  = threshold_data["threshold"]
    MODEL_TYPE = threshold_data.get("model_type", "unknown")

# ThreadPoolExecutor for parallel inference
# sklearn/LightGBM release the GIL during C-level computation
# This enables TRUE parallelism via threads
_executor = ThreadPoolExecutor(max_workers=4)

print(f"✅ Model loaded: {MODEL_TYPE}")
print(f"✅ Threshold   : {THRESHOLD:.4f}")
print(f"✅ Features    : {len(FEATURE_COLS)}")


def score_transaction(features: dict) -> float:
    """
    Returns fraud risk score between 0 and 1.
    Uses DataFrame to preserve feature names (no sklearn warnings).
    """
    df   = pd.DataFrame([features], columns=FEATURE_COLS)
    risk = model.predict_proba(df)[0][1]
    return float(risk)


def score_transaction_async(features: dict):
    """
    Submit inference to thread pool.
    Returns a Future object — call .result() to get the score.
    Useful for parallel batch scoring.
    """
    return _executor.submit(score_transaction, features)


def score_batch(feature_list: list) -> list:
    """
    Score multiple transactions in parallel using ThreadPoolExecutor.
    LightGBM releases the GIL during C-level prediction,
    enabling true parallel execution.
    
    Example:
        scores = score_batch([tx1_features, tx2_features, tx3_features])
    """
    futures = [_executor.submit(score_transaction, f) for f in feature_list]
    return [future.result() for future in futures]


def decide(risk: float) -> str:
    """Converts risk score into business decision."""
    if risk < 0.25:
        return "APPROVE"
    elif risk < THRESHOLD:
        return "STEP_UP_AUTH"
    else:
        return "BLOCK"


def explain(features: dict, risk: float) -> list:
    """Returns human-readable fraud reasons."""
    reasons = []

    if features.get("tx_count_1min", 0) > 3:
        reasons.append("High transaction velocity in last 1 minute")
    if features.get("tx_count_10min", 0) > 10:
        reasons.append("Unusual frequency in last 10 minutes")
    if features.get("is_night", 0) == 1:
        reasons.append("Transaction occurred at night")
    if features.get("Amount", 0) > 2000:
        reasons.append("Unusually high transaction amount")
    if features.get("amount_deviation", 0) > 3:
        reasons.append("Amount deviates significantly from recent average")
    if risk > THRESHOLD:
        reasons.append(f"Strong fraud pattern detected by {MODEL_TYPE} model")

    if not reasons:
        reasons.append("Transaction behavior within normal limits")

    return reasons


def get_model_info() -> dict:
    """Returns model metadata for health checks."""
    return {
        "model_type": MODEL_TYPE,
        "threshold":  THRESHOLD,
        "features":   len(FEATURE_COLS),
    }


# ── Quick Test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  INFERENCE ENGINE TEST")
    print("="*50)

    # Build test transaction
    test_features = {f"V{i}": 0.0 for i in range(1, 29)}
    test_features.update({
        "Amount":                2500.0,
        "amount_log":            np.log1p(2500.0),
        "amount_sqrt":           np.sqrt(2500.0),
        "tx_count_1min":         4,
        "tx_count_10min":        12,
        "tx_count_60min":        20,
        "amount_rolling_mean_1h": 300.0,
        "amount_rolling_std_1h":  100.0,
        "amount_deviation":       22.0,
        "hour":                  2,
        "is_night":              1
    })

    # Single inference
    print("\n[1] Single inference:")
    start = time.time()
    risk  = score_transaction(test_features)
    t     = (time.time() - start) * 1000
    print(f"    Risk Score : {risk:.4f}")
    print(f"    Decision   : {decide(risk)}")
    print(f"    Latency    : {t:.2f}ms")
    print(f"    Reasons    : {explain(test_features, risk)}")

    # Batch inference (parallel)
    print("\n[2] Batch inference (10 transactions, parallel):")
    batch = [test_features.copy() for _ in range(10)]
    start = time.time()
    scores = score_batch(batch)
    t = (time.time() - start) * 1000
    print(f"    Scores     : {[round(s, 4) for s in scores]}")
    print(f"    Total time : {t:.2f}ms")
    print(f"    Per tx     : {t/10:.2f}ms")

    # Throughput estimate
    single_ms = t / 10
    rpm        = 60000 / single_ms
    print(f"\n[3] Throughput estimate:")
    print(f"    {rpm:,.0f} predictions/min (single worker)")
    print(f"    {rpm*8:,.0f} predictions/min (8 workers)")

    print("\n" + "="*50)
    print("  INFERENCE ENGINE READY ✅")
    print("="*50)
