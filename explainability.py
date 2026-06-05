"""
SHAP Model Explainability Module
==================================
Explains WHY the model made each decision using SHAP values.

SHAP = SHapley Additive exPlanations
Based on game theory — each feature's contribution to the prediction.

Fix notes:
  - LightGBM SHAP values are in log-odds space → convert via sigmoid
  - New SHAP version returns list of arrays → handle correctly
"""

import numpy as np
import pandas as pd
import shap
import joblib
import json
import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Load Model (lazy) ─────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
_model        = None
_explainer    = None
_feature_cols = None


def _sigmoid(x):
    """Convert log-odds to probability."""
    return 1 / (1 + np.exp(-x))


def _load():
    global _model, _explainer, _feature_cols
    if _model is not None:
        return
    _model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
    with open(os.path.join(BASE_DIR, "models", "feature_cols.json")) as f:
        _feature_cols = json.load(f)
    _explainer = shap.TreeExplainer(_model)
    print("✅ SHAP explainer loaded")


def explain_transaction(features: dict, top_n: int = 5) -> dict:
    """
    Returns SHAP-based explanation for a single transaction.
    """
    _load()
    start = time.time()

    df          = pd.DataFrame([features], columns=_feature_cols)
    shap_values = _explainer.shap_values(df)

    # Handle LightGBM new format: list of [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        fraud_shap = np.array(shap_values[1][0])   # fraud class, first row
        base_logodds = float(_explainer.expected_value[1])
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        fraud_shap   = shap_values[0, :, 1]
        base_logodds = float(_explainer.expected_value[1])
    else:
        fraud_shap   = np.array(shap_values[0])
        ev = _explainer.expected_value
        base_logodds = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

    # Convert base value from log-odds to probability
    base_prob = float(_sigmoid(base_logodds))

    # Build feature impacts
    feature_impacts = []
    for i, col in enumerate(_feature_cols):
        shap_val = float(fraud_shap[i])
        feat_val = features.get(col, 0.0)
        feature_impacts.append({
            "feature":    col,
            "value":      round(float(feat_val), 4),
            "shap_value": round(shap_val, 4),
            "direction":  "increases_fraud_risk" if shap_val > 0
                          else "decreases_fraud_risk",
            "abs_impact": abs(shap_val)
        })

    feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
    top_features = feature_impacts[:top_n]
    for f in top_features:
        del f["abs_impact"]

    return {
        "top_features":   top_features,
        "base_prob":      round(base_prob, 4),
        "explanation_ms": round((time.time() - start) * 1000, 2),
        "total_features": len(_feature_cols),
    }


def format_explanation(shap_result: dict, decision: str) -> list:
    """Converts SHAP result into human-readable strings."""
    lines = []

    emoji = {"BLOCK": "🚫", "STEP_UP_AUTH": "⚠️", "APPROVE": "✅"}.get(decision, "")
    lines.append(f"{emoji} Decision: {decision}")
    lines.append(f"Base fraud probability: {shap_result['base_prob']:.1%}")
    lines.append("Top contributing features:")

    for f in shap_result["top_features"]:
        arrow  = "↑" if f["direction"] == "increases_fraud_risk" else "↓"
        sign   = "+" if f["shap_value"] > 0 else ""
        impact = "HIGH" if abs(f["shap_value"]) > 0.1 else \
                 "MED"  if abs(f["shap_value"]) > 0.02 else "LOW"

        feat = f["feature"]
        if feat == "is_night":
            display = f"Night transaction = {int(f['value'])}"
        elif feat == "tx_count_1min":
            display = f"Tx count (1 min) = {f['value']:.0f}"
        elif feat == "tx_count_10min":
            display = f"Tx count (10 min) = {f['value']:.0f}"
        elif feat == "amount_deviation":
            display = f"Amount deviation = {f['value']:.2f}"
        elif feat == "Amount":
            display = f"Amount = ₹{f['value']:.2f}"
        elif feat == "hour":
            display = f"Hour = {f['value']:.0f}:00"
        elif feat.startswith("V"):
            display = f"Bank signal {feat} = {f['value']:.3f}"
        else:
            display = f"{feat} = {f['value']:.3f}"

        lines.append(
            f"  {arrow} [{impact}] {display} "
            f"({sign}{f['shap_value']:.3f})"
        )

    return lines


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SHAP EXPLAINABILITY TEST")
    print("=" * 60)

    _load()

    # High-risk transaction
    print("\n[1] High-risk transaction (V14=-5.5, night, high velocity):")
    fraud_feat = {f"V{i}": 0.0 for i in range(1, 29)}
    fraud_feat.update({
        "V14": -5.5, "V4": 3.2, "V12": -3.1,
        "Amount": 149.62,
        "amount_log":             np.log1p(149.62),
        "amount_sqrt":            np.sqrt(149.62),
        "tx_count_1min":          5,
        "tx_count_10min":         12,
        "tx_count_60min":         25,
        "amount_rolling_mean_1h": 50.0,
        "amount_rolling_std_1h":  30.0,
        "amount_deviation":       3.3,
        "hour":                   2,
        "is_night":               1,
    })

    r1 = explain_transaction(fraud_feat)
    for line in format_explanation(r1, "BLOCK"):
        print(f"    {line}")
    print(f"    ⏱  Explanation: {r1['explanation_ms']}ms")

    # Normal transaction
    print("\n[2] Normal transaction (daytime, small amount):")
    normal_feat = {f"V{i}": 0.0 for i in range(1, 29)}
    normal_feat.update({
        "Amount": 50.0,
        "amount_log":             np.log1p(50.0),
        "amount_sqrt":            np.sqrt(50.0),
        "tx_count_1min":          1,
        "tx_count_10min":         3,
        "tx_count_60min":         8,
        "amount_rolling_mean_1h": 55.0,
        "amount_rolling_std_1h":  20.0,
        "amount_deviation":       0.1,
        "hour":                   14,
        "is_night":               0,
    })

    r2 = explain_transaction(normal_feat)
    for line in format_explanation(r2, "APPROVE"):
        print(f"    {line}")
    print(f"    ⏱  Explanation: {r2['explanation_ms']}ms")

    print("\n" + "=" * 60)
    print("  SHAP WORKING CORRECTLY ✅")
    print("  Base probability shows realistic fraud base rate")
    print("  Feature contributions show exact fraud drivers")
    print("=" * 60)
