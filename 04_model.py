import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("CHARGEMENT ET PREPARATION DES DONNEES")
print("=" * 50)

# 1. CHARGEMENT
sp500 = pd.read_csv('data/sp500_features.csv', index_col=0)
sp500.index = pd.to_datetime(sp500.index, format='mixed')

vix = pd.read_csv('data/vix_features.csv', index_col=0)
vix.index = pd.to_datetime(vix.index, format='mixed')

btc = pd.read_csv('data/btc_features.csv', index_col=0)
btc.index = pd.to_datetime(btc.index, format='mixed')

# 2. CONSTRUCTION DU DATASET
df = pd.DataFrame(index=sp500.index)
df['ret_sp500']  = sp500['rendement']
df['vol_sp500']  = sp500['vol_20j']
df['rsi_sp500']  = sp500['RSI']
df['macd_sp500'] = sp500['MACD']
df['ma_ratio_sp500'] = sp500['MA_ratio']

vix_aligned = vix.reindex(sp500.index, method='ffill')
df['vix_niveau'] = vix_aligned['Close']
df['vix_vol']    = vix_aligned['vol_20j']
df['vix_rsi']    = vix_aligned['RSI']

df.dropna(inplace=True)

# 3. VARIABLE CIBLE (PREDICTION J+1)
def definir_regime(v):
    if v > 25: return 2   # DANGER
    elif v > 20: return 1 # STRESS
    else: return 0        # CALME

# shift(-1) pour prédire le régime de DEMAIN
df['regime'] = df['vix_niveau'].shift(-1).apply(definir_regime)
df.dropna(inplace=True)

features = ['ret_sp500','vol_sp500','rsi_sp500', 'macd_sp500','ma_ratio_sp500', 'vix_niveau','vix_vol','vix_rsi']
X = df[features]
y = df['regime']

# 4. SPLIT TEMPOREL
n = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

# 5. SCALING (ANTI-LEAKAGE)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# 6. RÉÉQUILIBRAGE AVEC SMOTE
# k_neighbors=1 pour éviter les erreurs si la classe DANGER est très petite
sm = SMOTE(random_state=42, k_neighbors=1)
X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)

print(f"Distribution après SMOTE : {pd.Series(y_train_res).value_counts().to_dict()}")

# 7. ENTRAÎNEMENT DU MODELE FINAL
modele = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
)

# On entraîne sur les données RÉÉQUILIBRÉES (X_train_res)
modele.fit(X_train_res, y_train_res)

# 8. EVALUATION SUR VALIDATION
print("\n" + "=" * 50)
print("EVALUATION SUR VALIDATION (S&P 500)")
print("=" * 50)
y_pred = modele.predict(X_val_sc)
print(classification_report(y_val, y_pred, target_names=['CALME', 'STRESS', 'DANGER']))

# 9. WALK-FORWARD VALIDATION (ROBUSTESSE)
print("\nWALK-FORWARD VALIDATION EN COURS...")
tscv = TimeSeriesSplit(n_splits=5)
wf_scores = []

for i, (t_idx, v_idx) in enumerate(tscv.split(X_train_sc)):
    X_wf_tr, X_wf_ts = X_train_sc[t_idx], X_train_sc[v_idx]
    y_wf_tr, y_wf_ts = y_train.iloc[t_idx], y_train.iloc[v_idx]
    
    # On applique SMOTE sur chaque pli (fold) pour être rigoureux
    sm_wf = SMOTE(random_state=42, k_neighbors=1)
    X_wf_res, y_wf_res = sm_wf.fit_resample(X_wf_tr, y_wf_tr)
    
    m_wf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')
    m_wf.fit(X_wf_res, y_wf_res)
    wf_scores.append(m_wf.score(X_wf_ts, y_wf_ts))
    print(f"Fold {i+1} : {wf_scores[-1]:.3f}")

# 10. TEST DE GENERALISATION (BITCOIN)
print("\n" + "=" * 50)
print("TEST DE GENERALISATION SUR BITCOIN")
print("=" * 50)

df_btc = pd.DataFrame(index=btc.index)
df_btc['ret_sp500']      = btc['rendement']
df_btc['vol_sp500']      = btc['vol_20j']
df_btc['rsi_sp500']      = btc['RSI']
df_btc['macd_sp500']     = btc['MACD']
df_btc['ma_ratio_sp500'] = btc['MA_ratio']

# VIX adapté au BTC
vol_ann = btc['vol_20j'] * np.sqrt(252)
df_btc['vix_niveau'] = vol_ann.rolling(10).mean() * 100
df_btc['vix_vol']    = df_btc['vix_niveau'].rolling(20).std()
df_btc['vix_rsi']    = btc['RSI'] # On réutilise le RSI du BTC
df_btc.dropna(inplace=True)

df_btc['regime_reel'] = df_btc['vix_niveau'].apply(definir_regime)

X_btc_sc = scaler.transform(df_btc[features])
y_pred_btc = modele.predict(X_btc_sc)

print(classification_report(df_btc['regime_reel'], y_pred_btc, target_names=['CALME', 'STRESS', 'DANGER']))

print("\nMODELE FINAL PRET ET ROBUSTE 🚀")