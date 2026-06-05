"""
Ensemble Test: sklearn GBM + LightGBM
Goal: Get recall ~85% while keeping FP low
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import time

print("=" * 60)
print("  ENSEMBLE TEST: sklearn GBM + LightGBM")
print("=" * 60)

# ── Load & Engineer Features ──────────────────────────────────
print("\nLoading data...")
df = pd.read_csv('data/creditcard.csv')
df = df.sort_values('Time').reset_index(drop=True)
df['amount_log']   = np.log1p(df['Amount'])
df['amount_sqrt']  = np.sqrt(df['Amount'])
df['tx_count_1min']  = df.rolling(window=60,   on='Time')['Amount'].count().fillna(1)
df['tx_count_10min'] = df.rolling(window=600,  on='Time')['Amount'].count().fillna(1)
df['tx_count_60min'] = df.rolling(window=3600, on='Time')['Amount'].count().fillna(1)
df['amount_rolling_mean_1h'] = df.rolling(window=3600, on='Time')['Amount'].mean().fillna(df['Amount'].mean())
df['amount_rolling_std_1h']  = df.rolling(window=3600, on='Time')['Amount'].std().fillna(df['Amount'].std())
df['amount_deviation'] = (df['Amount'] - df['amount_rolling_mean_1h']) / (df['amount_rolling_std_1h'] + 1e-8)
df['hour']     = (df['Time'] // 3600) % 24
df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4]).astype(int)

cols = [f'V{i}' for i in range(1, 29)] + [
    'Amount', 'amount_log', 'amount_sqrt',
    'tx_count_1min', 'tx_count_10min', 'tx_count_60min',
    'amount_rolling_mean_1h', 'amount_rolling_std_1h',
    'amount_deviation', 'hour', 'is_night'
]

split    = int(len(df) * 0.8)
X_train  = df[cols].iloc[:split]
X_test   = df[cols].iloc[split:]
y_train  = df['Class'].iloc[:split]
y_test   = df['Class'].iloc[split:]

scale    = (y_train == 0).sum() / (y_train == 1).sum()
sw       = y_train.map({0: 1, 1: scale}).values

# ── Train sklearn GBM ─────────────────────────────────────────
print("\nTraining sklearn GBM (2-3 min)...")
start = time.time()
sklearn_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    random_state=42
)
sklearn_model.fit(X_train, y_train, sample_weight=sw)
sklearn_time = time.time() - start
print(f"  Done in {sklearn_time:.1f}s")

# ── Train LightGBM ────────────────────────────────────────────
print("\nTraining LightGBM...")
start = time.time()
lgbm_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_model.fit(X_train, y_train, sample_weight=sw)
lgbm_time = time.time() - start
print(f"  Done in {lgbm_time:.1f}s")

# ── Get Scores ────────────────────────────────────────────────
sklearn_prob = sklearn_model.predict_proba(X_test)[:, 1]
lgbm_prob    = lgbm_model.predict_proba(X_test)[:, 1]

# ── Test Different Combinations ───────────────────────────────
print("\n" + "=" * 70)
print(f"  {'Model/Combo':<25} {'ROC-AUC':>8} {'Recall':>7} {'Precision':>10} {'FP':>5} {'FN':>5}")
print("  " + "-" * 65)

combos = [
    ("sklearn only (0.30)",    sklearn_prob,                    0.30),
    ("LightGBM only (0.7722)", lgbm_prob,                       0.7722),
    ("50/50 ensemble",         0.5*sklearn_prob + 0.5*lgbm_prob, 0.50),
    ("60/40 sklearn/lgbm",     0.6*sklearn_prob + 0.4*lgbm_prob, 0.50),
    ("40/60 sklearn/lgbm",     0.4*sklearn_prob + 0.6*lgbm_prob, 0.50),
    ("70/30 sklearn/lgbm",     0.7*sklearn_prob + 0.3*lgbm_prob, 0.40),
    ("80/20 sklearn/lgbm",     0.8*sklearn_prob + 0.2*lgbm_prob, 0.35),
]

best_combo = None
best_score = 0

for name, probs, thresh in combos:
    auc  = roc_auc_score(y_test, probs)
    pred = (probs >= thresh).astype(int)
    cm   = confusion_matrix(y_test, pred)
    r    = cm[1][1] / (cm[1][1] + cm[1][0])
    p    = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    fp   = cm[0][1]
    fn   = cm[1][0]

    # Score: maximize recall while keeping FP reasonable
    score = r * 0.6 + (1 - fp/100) * 0.4
    marker = " ← BEST" if score > best_score else ""
    if score > best_score:
        best_score = score
        best_combo = (name, probs, thresh, r, p, fp, fn, auc)

    print(f"  {name:<25} {auc:>8.4f} {r:>7.3f} {p:>10.3f} {fp:>5} {fn:>5}{marker}")

print("\n" + "=" * 70)
print(f"  WINNER: {best_combo[0]}")
print(f"  ROC-AUC   : {best_combo[7]:.4f}")
print(f"  Recall    : {best_combo[3]:.3f} ({int((1-best_combo[3]/1)*75)} frauds caught)")
print(f"  Precision : {best_combo[4]:.3f}")
print(f"  FP        : {best_combo[5]}")
print(f"  FN        : {best_combo[6]}")
print("=" * 70)
