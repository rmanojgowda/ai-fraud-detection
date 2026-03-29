import joblib
import numpy as np
import pandas as pd
import json
import os

# ── Load model artifacts once at startup ─────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model     = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
with open(os.path.join(BASE_DIR, "models", "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)
with open(os.path.join(BASE_DIR, "models", "threshold.json")) as f:
    THRESHOLD = json.load(f)["threshold"]


def score_transaction(features: dict) -> float:
    """Returns fraud risk score between 0 and 1."""
    df = pd.DataFrame([features], columns=FEATURE_COLS)
    risk = model.predict_proba(df)[0][1]
    return float(risk)


def decide(risk: float) -> str:
    """Converts risk score into business decision."""
    if risk < 0.25:
        return "APPROVE"
    elif risk < 0.6:
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
    if risk > 0.6:
        reasons.append("Strong fraud pattern detected by ML model")
    if not reasons:
        reasons.append("Transaction behavior within normal limits")

    return reasons


if __name__ == "__main__":
    # Quick test
    test_features = {f"V{i}": 0.0 for i in range(1, 29)}
    test_features.update({
        "Amount": 2500,
        "amount_log": np.log1p(2500),
        "tx_count_1min": 4,
        "tx_count_10min": 12,
        "tx_count_60min": 20,
        "amount_rolling_mean_1h": 300,
        "hour": 2,
        "is_night": 1
    })

    risk     = score_transaction(test_features)
    decision = decide(risk)
    reasons  = explain(test_features, risk)

    print(f"Risk Score : {risk:.4f}")
    print(f"Decision   : {decision}")
    print(f"Reasons    : {reasons}")
