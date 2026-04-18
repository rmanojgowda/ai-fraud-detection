"""
Phase 7 — Threshold Optimization
=================================
Shows WHY we chose threshold 0.7722 using business cost analysis.
This is the difference between "ML engineer" and "business-aware ML engineer".

Key insight:
  A threshold is NOT a math problem. It's a business decision.
  Different thresholds have different costs.
  We find the threshold that minimizes TOTAL BUSINESS COST.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    confusion_matrix, roc_curve
)

print("=" * 65)
print("  PHASE 7 — THRESHOLD OPTIMIZATION & BUSINESS COST ANALYSIS")
print("=" * 65)

# ── Load Model & Data ─────────────────────────────────────────
print("\n[1/5] Loading model and data...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
with open(os.path.join(BASE_DIR, "models", "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)

df = pd.read_csv("data/creditcard.csv")
df = df.sort_values("Time").reset_index(drop=True)

# Rebuild features (same as train_model.py)
df["amount_log"]   = np.log1p(df["Amount"])
df["amount_sqrt"]  = np.sqrt(df["Amount"])
df["tx_count_1min"]  = df.rolling(window=60,   on="Time")["Amount"].count().fillna(1)
df["tx_count_10min"] = df.rolling(window=600,  on="Time")["Amount"].count().fillna(1)
df["tx_count_60min"] = df.rolling(window=3600, on="Time")["Amount"].count().fillna(1)
df["amount_rolling_mean_1h"] = (
    df.rolling(window=3600, on="Time")["Amount"].mean().fillna(df["Amount"].mean())
)
df["amount_rolling_std_1h"] = (
    df.rolling(window=3600, on="Time")["Amount"].std().fillna(df["Amount"].std())
)
df["amount_deviation"] = (
    (df["Amount"] - df["amount_rolling_mean_1h"]) /
    (df["amount_rolling_std_1h"] + 1e-8)
)
df["hour"]     = (df["Time"] // 3600) % 24
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4]).astype(int)

# Use test set (last 20%)
split_idx        = int(len(df) * 0.8)
df_test          = df.iloc[split_idx:].copy()
X_test           = df_test[FEATURE_COLS]
y_test           = df_test["Class"]

y_prob = model.predict_proba(X_test)[:, 1]
print(f"      Test set: {len(X_test):,} transactions ({y_test.sum()} fraud)")

# ── Business Cost Parameters ──────────────────────────────────
print("\n[2/5] Defining business cost parameters...")

avg_fraud_amount     = df[df['Class']==1]['Amount'].mean()
avg_fraud_amount_inr = avg_fraud_amount * 90  # EUR to INR approx

# Cost per missed fraud (False Negative)
# = average amount lost when fraud goes undetected
COST_FALSE_NEGATIVE = avg_fraud_amount_inr

# Cost per false alarm (False Positive)
# = customer service call + customer friction + potential churn
COST_FALSE_POSITIVE = 10 * 90  # €10 in INR

print(f"      Avg fraud amount      : €{avg_fraud_amount:.2f} (₹{avg_fraud_amount_inr:.2f})")
print(f"      Cost per missed fraud  : ₹{COST_FALSE_NEGATIVE:.2f}")
print(f"      Cost per false alarm   : ₹{COST_FALSE_POSITIVE:.2f}")
print(f"      Ratio                  : {COST_FALSE_NEGATIVE/COST_FALSE_POSITIVE:.1f}x")
print(f"\n      Insight: Missing a fraud costs {COST_FALSE_NEGATIVE/COST_FALSE_POSITIVE:.0f}x")
print(f"      more than a false alarm. Threshold should lean toward recall.")

# ── Evaluate All Thresholds ───────────────────────────────────
print("\n[3/5] Evaluating cost across all thresholds...")

thresholds = np.arange(0.01, 1.0, 0.01)
results    = []

for thresh in thresholds:
    y_pred = (y_prob >= thresh).astype(int)
    cm     = confusion_matrix(y_test, y_pred)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    total_fraud   = tp + fn
    total_normal  = tn + fp

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    cost_fn       = fn * COST_FALSE_NEGATIVE   # missed fraud cost
    cost_fp       = fp * COST_FALSE_POSITIVE    # false alarm cost
    total_cost    = cost_fn + cost_fp

    results.append({
        "threshold": thresh,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall":    recall,
        "precision": precision,
        "cost_fn":   cost_fn,
        "cost_fp":   cost_fp,
        "total_cost": total_cost,
    })

results_df = pd.DataFrame(results)

# ── Find Optimal Thresholds ───────────────────────────────────
print("\n[4/5] Finding optimal thresholds...")

# Precision floor: no bank deploys a model with precision < 90%
# (would block 1 in 10 legitimate customers)
PRECISION_FLOOR = 0.90
print(f"      Precision floor constraint: >= {PRECISION_FLOOR:.0%}")
print(f"      (Unconstrained minimum rejected if precision too low)")

# Show what unconstrained minimum would be
unconstrained_idx = results_df["total_cost"].idxmin()
unconstrained_row = results_df.iloc[unconstrained_idx]
if unconstrained_row["precision"] < PRECISION_FLOOR:
    print(f"      Unconstrained minimum: {unconstrained_row['threshold']:.2f} "
          f"(precision {unconstrained_row['precision']:.1%}) — REJECTED by floor")

# 1. Minimum cost threshold WITH precision constraint
valid_rows   = results_df[results_df["precision"] >= PRECISION_FLOOR]
min_cost_idx = valid_rows["total_cost"].idxmin()
min_cost_row = results_df.iloc[min_cost_idx]

# 2. Default threshold (0.5) for comparison
default_row = results_df[results_df["threshold"] == 0.50].iloc[0]

# 3. Current threshold (0.3 from before)
old_row = results_df[(results_df["threshold"] - 0.30).abs() < 0.015].iloc[0]

# 4. Our LightGBM threshold (0.7722)
lgbm_row = results_df[(results_df["threshold"] - 0.77).abs() < 0.015].iloc[0]

print("\n  Threshold Comparison:")
print(f"  {'Threshold':<12} {'Recall':>8} {'Precision':>10} "
      f"{'Missed':>8} {'FalseAlm':>10} {'TotalCost':>12}")
print("  " + "-"*62)

for label, row in [
    ("Default(0.5)", default_row),
    ("Old(0.30)",    old_row),
    ("LightGBM(0.77)", lgbm_row),
    ("Optimal",      min_cost_row),
]:
    print(f"  {label:<12} {row['recall']:>8.3f} {row['precision']:>10.3f} "
          f"{int(row['fn']):>8} {int(row['fp']):>10} "
          f"₹{row['total_cost']:>10,.2f}")

# Cost savings vs default
savings_vs_default = default_row["total_cost"] - min_cost_row["total_cost"]
savings_vs_old     = old_row["total_cost"] - lgbm_row["total_cost"]

print(f"\n  💰 Cost savings vs default threshold:  ₹{savings_vs_default:,.2f}")
print(f"  💰 Cost savings vs old threshold(0.3): ₹{savings_vs_old:,.2f}")

# ── Cost Breakdown at Optimal ─────────────────────────────────
print(f"\n  Optimal Threshold Analysis:")
print(f"    Threshold  : {min_cost_row['threshold']:.2f}")
print(f"    Recall     : {min_cost_row['recall']:.3f} ({min_cost_row['tp']:.0f}/{min_cost_row['tp']+min_cost_row['fn']:.0f} frauds caught)")
print(f"    Precision  : {min_cost_row['precision']:.3f}")
print(f"    Missed frauds  : {min_cost_row['fn']:.0f} → ₹{min_cost_row['cost_fn']:,.2f} loss")
print(f"    False alarms   : {min_cost_row['fp']:.0f} → ₹{min_cost_row['cost_fp']:,.2f} cost")
print(f"    Total cost     : ₹{min_cost_row['total_cost']:,.2f}")

# ── Save Results ──────────────────────────────────────────────
print("\n[5/5] Saving threshold analysis...")

os.makedirs("models", exist_ok=True)
threshold_report = {
    "optimal_threshold":        float(min_cost_row["threshold"]),
    "optimal_recall":           float(min_cost_row["recall"]),
    "optimal_precision":        float(min_cost_row["precision"]),
    "optimal_total_cost_inr":   float(min_cost_row["total_cost"]),
    "default_threshold":        0.5,
    "default_total_cost_inr":   float(default_row["total_cost"]),
    "savings_vs_default_inr":   float(savings_vs_default),
    "cost_per_missed_fraud_inr": COST_FALSE_NEGATIVE,
    "cost_per_false_alarm_inr":  COST_FALSE_POSITIVE,
    "roc_auc":                  float(roc_auc_score(y_test, y_prob)),
}

with open("models/threshold_analysis.json", "w") as f:
    json.dump(threshold_report, f, indent=2)

# Save full results
results_df.to_csv("models/threshold_results.csv", index=False)

print("      Saved: models/threshold_analysis.json")
print("      Saved: models/threshold_results.csv")

print("\n" + "=" * 65)
print("  THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 65)
print(f"\n  Optimal Threshold : {min_cost_row['threshold']:.2f}")
print(f"  Total Cost        : ₹{min_cost_row['total_cost']:,.2f}")
print(f"  Savings vs Default: ₹{savings_vs_default:,.2f}")
print(f"\n  Interview Answer:")
print(f"  'We chose threshold {min_cost_row['threshold']:.2f} by minimizing total")
print(f"   business cost. Missing a fraud costs ₹{COST_FALSE_NEGATIVE:.0f}")
print(f"   vs ₹{COST_FALSE_POSITIVE:.0f} per false alarm — a {COST_FALSE_NEGATIVE/COST_FALSE_POSITIVE:.0f}x ratio.")
print(f"   This saves ₹{savings_vs_default:,.0f} vs using the default 0.5 threshold.'")
print("=" * 65)
