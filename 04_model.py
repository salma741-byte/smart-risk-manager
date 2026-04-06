import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("CHARGEMENT ET PREPARATION DES DONNEES")
print("=" * 50)

# ─────────────────────────────────────────
# 1. CHARGEMENT DEPUIS SQLITE
# ─────────────────────────────────────────
conn  = sqlite3.connect('data/market_data.db')
sp500 = pd.read_sql("SELECT * FROM sp500_features",
                    conn, index_col='date', parse_dates=['date'])
vix   = pd.read_sql("SELECT * FROM vix_features",
                    conn, index_col='date', parse_dates=['date'])
btc   = pd.read_sql("SELECT * FROM btc_features",
                    conn, index_col='date', parse_dates=['date'])
conn.close()

print(f"S&P500 : {len(sp500)} jours")
print(f"VIX    : {len(vix)} jours")
print(f"BTC    : {len(btc)} jours")

# ─────────────────────────────────────────
# 2. CONSTRUCTION DU DATASET
# ─────────────────────────────────────────
df = pd.DataFrame(index=sp500.index)
df['ret_sp500']      = sp500['rendement']
df['vol_sp500']      = sp500['vol_20j']
df['rsi_sp500']      = sp500['rsi']
df['macd_sp500']     = sp500['macd']
df['ma_ratio_sp500'] = sp500['ma_ratio']

vix_aligned      = vix.reindex(sp500.index, method='ffill')
df['vix_niveau'] = vix_aligned['close']
df['vix_vol']    = vix_aligned['vol_20j']
df['vix_rsi']    = vix_aligned['rsi']
df.dropna(inplace=True)

# ─────────────────────────────────────────
# 3. VARIABLE CIBLE — PREDICTION J+1
# ─────────────────────────────────────────
def definir_regime(v):
    if v > 25:   return 2   # DANGER
    elif v > 20: return 1   # STRESS
    else:        return 0   # CALME

# shift(-1) = on predit le regime de DEMAIN
df['regime'] = df['vix_niveau'].shift(-1).apply(definir_regime)
df.dropna(inplace=True)

features = [
    'ret_sp500', 'vol_sp500', 'rsi_sp500',
    'macd_sp500', 'ma_ratio_sp500',
    'vix_niveau', 'vix_vol', 'vix_rsi'
]

X = df[features]
y = df['regime']

# ─────────────────────────────────────────
# 4. SPLIT TEMPOREL
# ─────────────────────────────────────────
n         = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val   = X.iloc[train_end:val_end]
X_test  = X.iloc[val_end:]
y_train = y.iloc[:train_end]
y_val   = y.iloc[train_end:val_end]
y_test  = y.iloc[val_end:]

print(f"\nSplit temporel :")
print(f"  Train : {len(X_train)} jours ({X_train.index[0].date()} → {X_train.index[-1].date()})")
print(f"  Val   : {len(X_val)} jours ({X_val.index[0].date()} → {X_val.index[-1].date()})")
print(f"  Test  : {len(X_test)} jours ({X_test.index[0].date()} → {X_test.index[-1].date()})")

# ─────────────────────────────────────────
# 5. SCALING ANTI-LEAKAGE
# ─────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 6. SMOTE — REEQUILIBRAGE
# ─────────────────────────────────────────
sm = SMOTE(random_state=42, k_neighbors=1)
X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)

print(f"\nDistribution apres SMOTE :")
print(pd.Series(y_train_res).value_counts().to_dict())

# ─────────────────────────────────────────
# 7. ENTRAINEMENT
# ─────────────────────────────────────────
modele = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
)
modele.fit(X_train_res, y_train_res)
print("\nModele entraine !")

# ─────────────────────────────────────────
# 8. EVALUATION SUR VALIDATION
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("EVALUATION SUR VALIDATION (S&P500)")
print("=" * 50)

y_pred = modele.predict(X_val_sc)
print(classification_report(
    y_val, y_pred,
    target_names=['CALME', 'STRESS', 'DANGER']
))

# ─────────────────────────────────────────
# 9. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────
print("WALK-FORWARD VALIDATION EN COURS...")
tscv      = TimeSeriesSplit(n_splits=5)
wf_scores = []

for i, (t_idx, v_idx) in enumerate(tscv.split(X_train_sc)):
    X_wf_tr = X_train_sc[t_idx]
    X_wf_ts = X_train_sc[v_idx]
    y_wf_tr = y_train.iloc[t_idx]
    y_wf_ts = y_train.iloc[v_idx]

    sm_wf = SMOTE(random_state=42, k_neighbors=1)
    X_wf_res, y_wf_res = sm_wf.fit_resample(X_wf_tr, y_wf_tr)

    m_wf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced'
    )
    m_wf.fit(X_wf_res, y_wf_res)
    score = m_wf.score(X_wf_ts, y_wf_ts)
    wf_scores.append(score)
    print(f"  Fold {i+1} : {score:.3f}")

print(f"  Moyenne : {np.mean(wf_scores):.3f}")

# ─────────────────────────────────────────
# 10. TEST GENERALISATION BITCOIN
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST DE GENERALISATION SUR BITCOIN")
print("=" * 50)

df_btc = pd.DataFrame(index=btc.index)
df_btc['ret_sp500']      = btc['rendement']
df_btc['vol_sp500']      = btc['vol_20j']
df_btc['rsi_sp500']      = btc['rsi']
df_btc['macd_sp500']     = btc['macd']
df_btc['ma_ratio_sp500'] = btc['ma_ratio']

vol_ann              = btc['vol_20j'] * np.sqrt(252)
df_btc['vix_niveau'] = vol_ann.rolling(10).mean() * 100
df_btc['vix_vol']    = df_btc['vix_niveau'].rolling(20).std()
df_btc['vix_rsi']    = btc['rsi']
df_btc.dropna(inplace=True)

df_btc['regime_reel'] = df_btc['vix_niveau'].apply(definir_regime)

X_btc_sc   = scaler.transform(df_btc[features])
y_pred_btc = modele.predict(X_btc_sc)

print(classification_report(
    df_btc['regime_reel'],
    y_pred_btc,
    target_names=['CALME', 'STRESS', 'DANGER']
))

# ─────────────────────────────────────────
# 11. SAUVEGARDER LES PREDICTIONS DANS SQLITE
# ─────────────────────────────────────────
proba_btc  = modele.predict_proba(X_btc_sc)
confiance  = proba_btc.max(axis=1)

predictions = pd.DataFrame({
    'date'         : df_btc.index.astype(str),
    'regime_predit': y_pred_btc,
    'regime_reel'  : df_btc['regime_reel'].values,
    'confiance'    : confiance,
    'vix_synthetique': df_btc['vix_niveau'].values
})

conn = sqlite3.connect('data/market_data.db')
predictions.to_sql('model_predictions', conn,
                   if_exists='replace', index=False)
conn.close()

print("\nPredictions sauvegardees → SQLite table 'model_predictions'")
print("\nMODELE FINAL PRET ET ROBUSTE")