import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import os
import json

print("=" * 50)
print("FRAUD DETECTION - MODEL TRAINING")
print("=" * 50)

# ── 1. Load Dataset ──────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv("data/creditcard.csv")
print(f"      Rows: {len(df):,} | Columns: {len(df.columns)}")
print(f"      Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")

# ── 2. Feature Engineering ───────────────────────
print("\n[2/5] Engineering features...")

df = df.sort_values("Time").reset_index(drop=True)

df["amount_log"] = np.log1p(df["Amount"])
df["tx_count_1min"]  = df.rolling(window=60,   on="Time")["Amount"].count().fillna(1)
df["tx_count_10min"] = df.rolling(window=600,  on="Time")["Amount"].count().fillna(1)
df["tx_count_60min"] = df.rolling(window=3600, on="Time")["Amount"].count().fillna(1)
df["amount_rolling_mean_1h"] = (
    df.rolling(window=3600, on="Time")["Amount"]
    .mean().fillna(df["Amount"].mean())
)
df["hour"]     = (df["Time"] // 3600) % 24
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4]).astype(int)

v_features = [f"V{i}" for i in range(1, 29)]
print("      Features created: V1-V28, amount, velocity, time-of-day")

# ── 3. Prepare ML Dataset ────────────────────────
print("\n[3/5] Preparing train/test split...")

FEATURE_COLS = v_features + [
    "Amount", "amount_log",
    "tx_count_1min", "tx_count_10min", "tx_count_60min",
    "amount_rolling_mean_1h", "hour", "is_night"
]

X = df[FEATURE_COLS]
y = df["Class"]

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"      Fraud in train: {y_train.sum()} | Fraud in test: {y_test.sum()}")

# ── 4. Train Model ───────────────────────────────
print("\n[4/5] Training Gradient Boosting model...")
print("      This will take 2-3 minutes...")

fraud_count  = y_train.sum()
normal_count = len(y_train) - fraud_count
scale        = normal_count / fraud_count
print(f"      Imbalance ratio: {scale:.1f}x — applying sample weights")

sample_weights = y_train.map({0: 1, 1: scale})

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=sample_weights)

y_prob    = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred    = (y_prob >= threshold).astype(int)
roc_auc   = roc_auc_score(y_test, y_prob)

print(f"\n      ROC-AUC Score : {roc_auc:.4f}")
print(f"      Threshold     : {threshold}")
print("\n      Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("      Confusion Matrix:")
print(f"        True Negatives : {cm[0][0]:,}")
print(f"        False Positives: {cm[0][1]:,}")
print(f"        False Negatives: {cm[1][0]:,}")
print(f"        True Positives : {cm[1][1]:,}")

print("\n      Top 5 Important Features:")
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
for feat, imp in importances.sort_values(ascending=False).head(5).items():
    print(f"        {feat}: {imp:.4f}")

# ── 5. Save ──────────────────────────────────────
print("\n[5/5] Saving model artifacts...")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")

with open("models/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)
with open("models/threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f)

print("      Saved: models/fraud_model.pkl")
print("      Saved: models/feature_cols.json")
print("      Saved: models/threshold.json")
print("\n" + "=" * 50)
print(f"TRAINING COMPLETE — ROC-AUC: {roc_auc:.4f}")
print("=" * 50)
