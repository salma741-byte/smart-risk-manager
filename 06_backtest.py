# ============================================================
#  MODULE 3 — GÉNÉRATION DES SIGNAUX & BACKTEST
#  Produit les signaux ACHAT / VENTE / NEUTRE
#  + backtest avec métriques de performance financière
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

DB_PATH  = "data/market_data.db"
MODEL_DIR = "models"
LSTM_LOOKBACK = 20

# ─────────────────────────────────────────
# 1. CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────
def charger_modeles():
    print("Chargement des modèles...")
    rf     = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    s_rf   = joblib.load(f"{MODEL_DIR}/scaler_rf.pkl")
    xgb_m  = xgb.XGBClassifier()
    xgb_m.load_model(f"{MODEL_DIR}/xgboost.json")
    lstm   = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_model.keras")
    s_lstm = joblib.load(f"{MODEL_DIR}/scaler_lstm.pkl")
    feats  = pd.read_csv(f"{MODEL_DIR}/feature_names.csv",
                          header=None)[0].tolist()
    poids  = np.load(f"{MODEL_DIR}/ensemble_weights.npy")
    print("  ✓ RF, XGBoost, LSTM chargés")
    return rf, s_rf, xgb_m, lstm, s_lstm, feats, poids

# ─────────────────────────────────────────
# 2. PRÉDICTION ENSEMBLE
# ─────────────────────────────────────────
def predire_proba_ensemble(X_raw, rf, s_rf, xgb_m,
                            lstm, s_lstm, feats, poids):
    """
    Retourne les probabilités ensemble pour chaque ligne de X_raw.
    Gère l'alignement LSTM (lookback).
    """
    n = len(X_raw)

    # RF
    X_rf   = s_rf.transform(X_raw)
    p_rf   = rf.predict_proba(X_rf)[:, 1]

    # XGBoost
    p_xgb  = xgb_m.predict_proba(X_raw)[:, 1]

    # LSTM — séquences
    X_lstm_s = s_lstm.transform(X_raw)
    p_lstm   = np.full(n, np.nan)
    if n >= LSTM_LOOKBACK:
        seqs = np.array([X_lstm_s[i-LSTM_LOOKBACK:i]
                         for i in range(LSTM_LOOKBACK, n)])
        raw_lstm = lstm.predict(seqs, verbose=0).flatten()
        p_lstm[LSTM_LOOKBACK:] = raw_lstm

    # Ensemble pondéré
    w_rf, w_xgb, w_lstm = poids
    valid  = ~np.isnan(p_lstm)
    probas = np.where(
        valid,
        w_rf * p_rf + w_xgb * p_xgb + w_lstm * p_lstm,
        (w_rf / (w_rf + w_xgb)) * p_rf +
        (w_xgb / (w_rf + w_xgb)) * p_xgb
    )
    return probas

# ─────────────────────────────────────────
# 3. RÈGLES DE SIGNAL
# ─────────────────────────────────────────
SEUIL_ACHAT  = 0.60   # probabilité hausse > 60% → ACHAT
SEUIL_VENTE  = 0.40   # probabilité hausse < 40% → VENTE
# Entre 40% et 60% → NEUTRE

def proba_vers_signal(probas):
    signaux = np.where(probas >= SEUIL_ACHAT, 'ACHAT',
              np.where(probas <= SEUIL_VENTE, 'VENTE', 'NEUTRE'))
    return signaux

# ─────────────────────────────────────────
# 4. BACKTEST
# ─────────────────────────────────────────
def backtest(df_feat, probas, signaux):
    """
    Stratégie long-only filtrée par signal.
    Position = 1 si signal ACHAT, 0 si NEUTRE/VENTE.
    Comparaison contre Buy & Hold.
    """
    conn = sqlite3.connect(DB_PATH)
    prix = pd.read_sql("SELECT date, close FROM sp500_prices",
                       conn, index_col='date', parse_dates=['date'])
    conn.close()

    bt = df_feat[[]].copy()
    bt = bt.join(prix['close'].rename('prix'))
    bt['proba']    = probas
    bt['signal']   = signaux
    bt['ret_bh']   = bt['prix'].pct_change()          # Buy & Hold
    bt['position'] = (bt['signal'] == 'ACHAT').astype(int).shift(1)
    bt['ret_strat'] = bt['position'] * bt['ret_bh']
    bt.dropna(inplace=True)

    # Métriques financières
    def sharpe(rets, rf=0.02/252):
        excess = rets - rf
        return (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0

    def max_drawdown(cum):
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        return dd.min()

    cum_bh    = (1 + bt['ret_bh']).cumprod()
    cum_strat = (1 + bt['ret_strat']).cumprod()

    metriques = {
        'Return_BH':       f"{(cum_bh.iloc[-1]-1)*100:.1f}%",
        'Return_Strat':    f"{(cum_strat.iloc[-1]-1)*100:.1f}%",
        'Sharpe_BH':       f"{sharpe(bt['ret_bh']):.2f}",
        'Sharpe_Strat':    f"{sharpe(bt['ret_strat']):.2f}",
        'MaxDD_BH':        f"{max_drawdown(cum_bh)*100:.1f}%",
        'MaxDD_Strat':     f"{max_drawdown(cum_strat)*100:.1f}%",
        'Win_Rate':        f"{(bt['ret_strat']>0).mean()*100:.1f}%",
        'Jours_investis':  f"{int(bt['position'].sum())} / {len(bt)}",
    }
    print("\n" + "=" * 55)
    print("  RÉSULTATS DU BACKTEST")
    print("=" * 55)
    for k, v in metriques.items():
        print(f"  {k:<20} {v}")

    return bt, metriques, cum_bh, cum_strat

# ─────────────────────────────────────────
# 5. SIGNAL DU JOUR (PRÉDICTION LIVE)
# ─────────────────────────────────────────
def signal_aujourdhui(rf, s_rf, xgb_m, lstm, s_lstm, feats, poids):
    """
    Charge les toutes dernières données et retourne
    le signal de trading pour le prochain jour de bourse.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sp500_ml_features",
                     conn, index_col='date', parse_dates=['date'])
    conn.close()
    df.sort_index(inplace=True)

    # Derniers LOOKBACK+1 jours
    X_recent = df[feats].values
    probas    = predire_proba_ensemble(X_recent, rf, s_rf, xgb_m,
                                       lstm, s_lstm, feats, poids)
    p_last    = probas[-1]
    signal    = proba_vers_signal(np.array([p_last]))[0]

    date_last = df.index[-1].strftime('%Y-%m-%d')
    print("\n" + "=" * 55)
    print(f"  SIGNAL POUR LE PROCHAIN JOUR DE BOURSE")
    print("=" * 55)
    print(f"  Dernière date dans les données : {date_last}")
    print(f"  Probabilité de HAUSSE J+1     : {p_last*100:.1f}%")
    print(f"  ──────────────────────────────────────")
    emoji = "🟢" if signal == 'ACHAT' else ("🔴" if signal == 'VENTE' else "🟡")
    print(f"  SIGNAL  →  {emoji}  {signal}")
    print(f"  Seuil achat  : >{SEUIL_ACHAT*100:.0f}%  |"
          f"  Seuil vente  : <{SEUIL_VENTE*100:.0f}%")
    print("=" * 55)
    return signal, p_last

# ─────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MODULE 3 — SIGNAUX & BACKTEST")
    print("=" * 55)

    rf, s_rf, xgb_m, lstm, s_lstm, feats, poids = charger_modeles()

    # Charger toutes les features
    conn = sqlite3.connect(DB_PATH)
    df_feat = pd.read_sql("SELECT * FROM sp500_ml_features",
                           conn, index_col='date', parse_dates=['date'])
    conn.close()
    df_feat.sort_index(inplace=True)

    X_all = df_feat[feats].values

    # Prédictions sur tout l'historique
    print("Calcul des probabilités sur tout l'historique...")
    probas  = predire_proba_ensemble(X_all, rf, s_rf, xgb_m,
                                      lstm, s_lstm, feats, poids)
    signaux = proba_vers_signal(probas)

    # Distribution des signaux
    unique, counts = np.unique(signaux, return_counts=True)
    print("\nDistribution des signaux :")
    for s, c in zip(unique, counts):
        print(f"  {s:<8} : {c} jours ({c/len(signaux)*100:.1f}%)")

    # Backtest
    bt, metrics, cum_bh, cum_strat = backtest(df_feat, probas, signaux)

    # Signal du jour
    signal_aujourdhui(rf, s_rf, xgb_m, lstm, s_lstm, feats, poids)

    # Export CSV des signaux historiques
    bt.to_csv("data/signaux_historiques.csv")
    print("\nSignaux exportés → data/signaux_historiques.csv")
    print("\nModule 3 terminé ✓")