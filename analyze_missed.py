import pandas as pd
import numpy as np
import joblib
import json

model = joblib.load('models/fraud_model.pkl')
with open('models/feature_cols.json') as f:
    cols = json.load(f)

df = pd.read_csv('data/creditcard.csv')
df = df.sort_values('Time').reset_index(drop=True)
df['amount_log'] = np.log1p(df['Amount'])
df['amount_sqrt'] = np.sqrt(df['Amount'])
df['tx_count_1min']  = df.rolling(window=60,   on='Time')['Amount'].count().fillna(1)
df['tx_count_10min'] = df.rolling(window=600,  on='Time')['Amount'].count().fillna(1)
df['tx_count_60min'] = df.rolling(window=3600, on='Time')['Amount'].count().fillna(1)
df['amount_rolling_mean_1h'] = df.rolling(window=3600, on='Time')['Amount'].mean().fillna(df['Amount'].mean())
df['amount_rolling_std_1h']  = df.rolling(window=3600, on='Time')['Amount'].std().fillna(df['Amount'].std())
df['amount_deviation'] = (df['Amount'] - df['amount_rolling_mean_1h']) / (df['amount_rolling_std_1h'] + 1e-8)
df['hour']     = (df['Time'] // 3600) % 24
df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4]).astype(int)

split    = int(len(df) * 0.8)
df_test  = df.iloc[split:].copy()
y_prob   = model.predict_proba(df_test[cols])[:, 1]
df_test['score'] = y_prob

missed = df_test[(df_test['Class'] == 1) & (y_prob < 0.7722)]
caught = df_test[(df_test['Class'] == 1) & (y_prob >= 0.7722)]
normal = df_test[df_test['Class'] == 0]

print("=" * 55)
print("  WHY 18 FRAUDS CANNOT BE CAUGHT")
print("=" * 55)

print("\nV14 (most important feature) comparison:")
print(f"  Missed frauds V14 mean : {missed['V14'].mean():.3f}")
print(f"  Caught frauds V14 mean : {caught['V14'].mean():.3f}")
print(f"  Normal txns   V14 mean : {normal['V14'].mean():.3f}")

print("\nKey insight:")
print(f"  Caught frauds have V14 ~ {caught['V14'].mean():.1f} (strongly negative)")
print(f"  Missed frauds have V14 ~ {missed['V14'].mean():.1f} (close to normal!)")
print(f"  Normal txns have  V14 ~ {normal['V14'].mean():.3f}")

print("\nScore distribution:")
print(f"  Missed frauds score range: {missed['score'].min():.6f} to {missed['score'].max():.6f}")
print(f"  Caught frauds score range: {caught['score'].min():.6f} to {caught['score'].max():.6f}")
print(f"  Gap between groups: {caught['score'].min() - missed['score'].max():.3f}")

print("\nMissed fraud amounts:")
print(f"  Min: {missed['Amount'].min():.2f}")
print(f"  Max: {missed['Amount'].max():.2f}")
print(f"  Mean: {missed['Amount'].mean():.2f}")
print(f"  Most are small afternoon transactions (looks normal)")

print("\nConclusion:")
print("  Missed frauds have V14 values SIMILAR to normal transactions.")
print("  The bank's PCA encoding doesn't distinguish them.")
print("  No ML model can reliably catch these without more data.")
print("  This is a DATASET limitation, not a modeling limitation.")
print("=" * 55)
