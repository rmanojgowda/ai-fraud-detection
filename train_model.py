import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
import joblib
import json
import os
import time

print("=" * 60)
print("  FRAUD DETECTION — PHASE 6 TRAINING (LightGBM)")
print("=" * 60)

# ── 1. Load Dataset ───────────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv("data/creditcard.csv")
print(f"      Rows     : {len(df):,}")
print(f"      Fraud    : {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

# ── 2. Feature Engineering ────────────────────────────────────
print("\n[2/6] Engineering features...")
df = df.sort_values("Time").reset_index(drop=True)

df["amount_log"]   = np.log1p(df["Amount"])
df["amount_sqrt"]  = np.sqrt(df["Amount"])

df["tx_count_1min"]  = df.rolling(window=60,   on="Time")["Amount"].count().fillna(1)
df["tx_count_10min"] = df.rolling(window=600,  on="Time")["Amount"].count().fillna(1)
df["tx_count_60min"] = df.rolling(window=3600, on="Time")["Amount"].count().fillna(1)

df["amount_rolling_mean_1h"] = (
    df.rolling(window=3600, on="Time")["Amount"]
    .mean().fillna(df["Amount"].mean())
)
df["amount_rolling_std_1h"] = (
    df.rolling(window=3600, on="Time")["Amount"]
    .std().fillna(df["Amount"].std())
)
df["amount_deviation"] = (
    (df["Amount"] - df["amount_rolling_mean_1h"]) /
    (df["amount_rolling_std_1h"] + 1e-8)
)

df["hour"]     = (df["Time"] // 3600) % 24
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4]).astype(int)

v_features   = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = v_features + [
    "Amount", "amount_log", "amount_sqrt",
    "tx_count_1min", "tx_count_10min", "tx_count_60min",
    "amount_rolling_mean_1h", "amount_rolling_std_1h",
    "amount_deviation", "hour", "is_night"
]
print(f"      Total features : {len(FEATURE_COLS)}")

# ── 3. Train/Test Split ───────────────────────────────────────
print("\n[3/6] Splitting dataset...")
X = df[FEATURE_COLS]
y = df["Class"]

split_idx        = int(len(X) * 0.8)
X_train, X_test  = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test  = y.iloc[:split_idx], y.iloc[split_idx:]

fraud_train  = y_train.sum()
normal_train = len(y_train) - fraud_train
scale        = normal_train / fraud_train

print(f"      Train : {len(X_train):,} ({fraud_train} fraud)")
print(f"      Test  : {len(X_test):,}  ({y_test.sum()} fraud)")
print(f"      Imbalance ratio : {scale:.1f}x")

# ── 4. Train LightGBM ─────────────────────────────────────────
print("\n[4/6] Training LightGBM model...")

start = time.time()

# Use sample weights instead of scale_pos_weight
# scale_pos_weight is too aggressive for early stopping
sample_weights = y_train.map({0: 1, 1: scale}).values

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Train WITHOUT early stopping first (more reliable)
model.fit(
    X_train, y_train,
    sample_weight=sample_weights
)

train_time = time.time() - start
print(f"      Training time : {train_time:.1f} seconds")
print(f"      Trees built   : {model.n_estimators_}")

# ── 5. Evaluate ───────────────────────────────────────────────
print("\n[5/6] Evaluating model...")

y_prob  = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

# Find optimal threshold using F2 score
# (weights recall 2x more than precision)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
beta   = 2
f_beta = ((1 + beta**2) * precisions * recalls) / \
         (beta**2 * precisions + recalls + 1e-8)

best_idx       = np.argmax(f_beta[:-1])  # exclude last point
best_threshold = float(thresholds[best_idx])
best_precision = float(precisions[best_idx])
best_recall    = float(recalls[best_idx])

print(f"\n      ROC-AUC Score         : {roc_auc:.4f}")
print(f"      Optimal Threshold     : {best_threshold:.4f}")
print(f"      Precision at thresh   : {best_precision:.4f}")
print(f"      Recall at thresh      : {best_recall:.4f}")

y_pred = (y_prob >= best_threshold).astype(int)
print("\n      Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("      Confusion Matrix:")
print(f"        True Negatives  : {cm[0][0]:,}")
print(f"        False Positives : {cm[0][1]:,}")
print(f"        False Negatives : {cm[1][0]:,}")
print(f"        True Positives  : {cm[1][1]:,}")

# Cost analysis
avg_fraud_amount     = df[df['Class']==1]['Amount'].mean()
cost_per_false_alarm = 10
total_fraud_loss     = cm[1][0] * avg_fraud_amount
total_false_alarm    = cm[0][1] * cost_per_false_alarm
total_cost           = total_fraud_loss + total_false_alarm

print(f"\n      Cost Analysis:")
print(f"        Avg fraud amount    : ₹{avg_fraud_amount:.2f}")
print(f"        Missed fraud loss   : ₹{total_fraud_loss:.2f}")
print(f"        False alarm cost    : ₹{total_false_alarm:.2f}")
print(f"        Total cost          : ₹{total_cost:.2f}")

# Top features
print("\n      Top 10 Important Features:")
importance = pd.Series(
    model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)
for feat, imp in importance.head(10).items():
    print(f"        {feat:<35}: {imp}")

# ── 6. Save ───────────────────────────────────────────────────
print("\n[6/6] Saving artifacts...")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")

with open("models/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)

with open("models/threshold.json", "w") as f:
    json.dump({
        "threshold":  best_threshold,
        "precision":  best_precision,
        "recall":     best_recall,
        "roc_auc":    roc_auc,
        "model_type": "LightGBM"
    }, f, indent=2)

print(f"      Saved: models/fraud_model.pkl")
print(f"      Saved: models/feature_cols.json")
print(f"      Saved: models/threshold.json")
print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  Threshold : {best_threshold:.4f}")
print(f"  Recall    : {best_recall:.4f}")
print(f"  Time      : {train_time:.1f}s")
print("=" * 60)
