"""
04_analyse.py — Analyse Statistique Quotidienne
================================================
Génère un rapport statistique complet sur le S&P500 et le VIX.
Stocke les résultats dans SQLite + export JSON pour dashboard.

Métriques calculées :
  • Statistiques descriptives (rendements)
  • Drawdown maximum glissant
  • Ratio de Sharpe annualisé (rolling 252j)
  • Skewness & Kurtosis (queues de distribution)
  • Corrélation SP500 / VIX
  • Régime de marché (Bull / Bear / Neutre)
  • Résumé du jour (signal qualitatif)
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DB_PATH      = "data/market_data.db"
OUTPUT_DIR   = "data/rapports"
RISK_FREE    = 0.05 / 252   # Taux sans risque journalier (~5% annuel US)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)


# ─────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
def charger_features(nom: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT * FROM {nom}_features ORDER BY date",
        conn, index_col='date', parse_dates=['date']
    )
    conn.close()
    return df


# ─────────────────────────────────────────
# CALCULS STATISTIQUES
# ─────────────────────────────────────────
def stats_descriptives(df: pd.DataFrame) -> dict:
    r = df['rendement'].dropna()
    return {
        "moyenne_journaliere":   round(r.mean(), 6),
        "moyenne_annualisee":    round(r.mean() * 252, 4),
        "volatilite_journaliere": round(r.std(), 6),
        "volatilite_annualisee":  round(r.std() * np.sqrt(252), 4),
        "min":    round(r.min(), 4),
        "max":    round(r.max(), 4),
        "median": round(r.median(), 6),
        "skewness":  round(r.skew(), 4),
        "kurtosis":  round(r.kurtosis(), 4),
        "nb_jours_positifs": int((r > 0).sum()),
        "nb_jours_negatifs": int((r < 0).sum()),
        "pct_jours_positifs": round((r > 0).mean() * 100, 2),
    }


def sharpe_ratio(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Sharpe ratio glissant annualisé."""
    r = df['rendement'].dropna()
    excess   = r - RISK_FREE
    rolling_mean = excess.rolling(window).mean()
    rolling_std  = r.rolling(window).std()
    sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    return sharpe


def max_drawdown(df: pd.DataFrame) -> dict:
    """Drawdown maximum sur l'historique complet + rolling 252j."""
    prix     = df['close']
    peak     = prix.cummax()
    drawdown = (prix - peak) / peak

    # Rolling 1 an
    rolling_dd = drawdown.rolling(252).min()

    return {
        "drawdown_max_global":     round(drawdown.min(), 4),
        "drawdown_max_1an":        round(rolling_dd.iloc[-1], 4),
        "date_drawdown_max_global": str(drawdown.idxmin().date()),
        "drawdown_actuel":         round(drawdown.iloc[-1], 4),
    }


def regime_marche(df: pd.DataFrame) -> dict:
    """
    Détermination du régime de marché basée sur :
    - MA50 vs MA200 (Golden Cross / Death Cross)
    - RSI
    - Volatilité 20j vs historique
    """
    derniere = df.iloc[-1]

    # MA200
    df['ma_200'] = df['close'].rolling(200).mean()
    ma200_actuel = df['ma_200'].iloc[-1]

    # Signal tendance
    if derniere['ma_20'] > derniere['ma_50'] > ma200_actuel:
        tendance = "BULL_FORT"
    elif derniere['ma_20'] > derniere['ma_50']:
        tendance = "BULL_MODERE"
    elif derniere['ma_20'] < derniere['ma_50'] < ma200_actuel:
        tendance = "BEAR_FORT"
    elif derniere['ma_20'] < derniere['ma_50']:
        tendance = "BEAR_MODERE"
    else:
        tendance = "NEUTRE"

    # Signal RSI
    rsi = derniere['rsi']
    if rsi > 70:
        signal_rsi = "SURACHETÉ"
    elif rsi < 30:
        signal_rsi = "SURVENDU"
    else:
        signal_rsi = "NEUTRE"

    # Signal volatilité
    vol_hist = df['vol_20j'].median()
    vol_actuelle = derniere['vol_20j']
    if vol_actuelle > vol_hist * 1.5:
        signal_vol = "ÉLEVÉE"
    elif vol_actuelle < vol_hist * 0.7:
        signal_vol = "FAIBLE"
    else:
        signal_vol = "NORMALE"

    # Signal MACD
    signal_macd = "HAUSSIER" if derniere['macd'] > derniere['macd_signal'] else "BAISSIER"

    return {
        "tendance":     tendance,
        "signal_rsi":   signal_rsi,
        "signal_vol":   signal_vol,
        "signal_macd":  signal_macd,
        "rsi_valeur":   round(rsi, 2),
        "ma_20":        round(derniere['ma_20'], 2),
        "ma_50":        round(derniere['ma_50'], 2),
        "ma_200":       round(ma200_actuel, 2),
        "close":        round(derniere['close'], 2),
        "rendement_j":  round(derniere['rendement'] * 100, 4),
        "vol_20j_ann":  round(derniere['vol_20j'] * np.sqrt(252) * 100, 2),
    }


def analyse_queues(df: pd.DataFrame) -> dict:
    """
    Value at Risk (VaR) et Expected Shortfall (CVaR).
    Standards en risk management institutionnel.
    """
    r = df['rendement'].dropna()

    var_95   = np.percentile(r, 5)
    var_99   = np.percentile(r, 1)
    cvar_95  = r[r <= var_95].mean()
    cvar_99  = r[r <= var_99].mean()

    return {
        "VaR_95_journalier":   round(var_95 * 100, 4),
        "VaR_99_journalier":   round(var_99 * 100, 4),
        "CVaR_95_journalier":  round(cvar_95 * 100, 4),
        "CVaR_99_journalier":  round(cvar_99 * 100, 4),
        "VaR_95_annualise":    round(var_95 * np.sqrt(252) * 100, 4),
        "rendement_max_1j":    round(r.max() * 100, 4),
        "rendement_min_1j":    round(r.min() * 100, 4),
    }


def analyse_rolling(df: pd.DataFrame) -> dict:
    """Indicateurs glissants récents : 5j, 1m, 3m, 1an."""
    close = df['close']
    today = close.iloc[-1]

    def perf(n):
        if len(close) > n:
            return round((today / close.iloc[-n] - 1) * 100, 2)
        return None

    sharpe = sharpe_ratio(df)

    return {
        "perf_5j":   perf(5),
        "perf_1m":   perf(21),
        "perf_3m":   perf(63),
        "perf_6m":   perf(126),
        "perf_1an":  perf(252),
        "perf_ytd":  round((today / df[df.index.year == datetime.today().year]['close'].iloc[0] - 1) * 100, 2) if not df[df.index.year == datetime.today().year].empty else None,
        "sharpe_1an_actuel": round(sharpe.iloc[-1], 4) if not sharpe.empty else None,
        "sharpe_moyen_historique": round(sharpe.mean(), 4),
    }


def correlation_sp500_vix(sp500: pd.DataFrame, vix: pd.DataFrame) -> dict:
    """Corrélation SP500/VIX — indicateur de stress de marché."""
    merged = pd.merge(
        sp500[['rendement']].rename(columns={'rendement': 'sp500'}),
        vix[['rendement']].rename(columns={'rendement': 'vix'}),
        left_index=True, right_index=True
    ).dropna()

    corr_global  = merged.corr().loc['sp500', 'vix']
    corr_1an     = merged.tail(252).corr().loc['sp500', 'vix']
    corr_3m      = merged.tail(63).corr().loc['sp500', 'vix']

    # Niveau de stress
    if corr_3m < -0.7:
        stress = "ÉLEVÉ — forte corrélation inverse (panique)"
    elif corr_3m < -0.5:
        stress = "MODÉRÉ"
    else:
        stress = "FAIBLE — marché calme"

    return {
        "correlation_globale":   round(corr_global, 4),
        "correlation_1an":       round(corr_1an, 4),
        "correlation_3m":        round(corr_3m, 4),
        "signal_stress":         stress,
        "nb_jours_communs":      len(merged),
    }


# ─────────────────────────────────────────
# RAPPORT COMPLET
# ─────────────────────────────────────────
def generer_rapport():
    print("=" * 55)
    print("ANALYSE STATISTIQUE QUOTIDIENNE — S&P500 / VIX")
    print(f"Date : {datetime.today().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 55)

    sp500 = charger_features('sp500')
    vix   = charger_features('vix')

    rapport = {
        "date_rapport":       datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
        "sp500": {
            "stats_descriptives": stats_descriptives(sp500),
            "drawdown":           max_drawdown(sp500),
            "regime":             regime_marche(sp500),
            "risque_queues":      analyse_queues(sp500),
            "performances":       analyse_rolling(sp500),
        },
        "vix": {
            "stats_descriptives": stats_descriptives(vix),
            "regime":             regime_marche(vix),
            "performances":       analyse_rolling(vix),
        },
        "correlation_sp500_vix": correlation_sp500_vix(sp500, vix),
    }

    # ── Affichage console
    print("\n📊 S&P500 — SITUATION DU JOUR")
    reg = rapport['sp500']['regime']
    print(f"  Clôture      : {reg['close']}")
    print(f"  Rendement J  : {reg['rendement_j']:+.4f}%")
    print(f"  Tendance     : {reg['tendance']}")
    print(f"  RSI          : {reg['rsi_valeur']} → {reg['signal_rsi']}")
    print(f"  MACD         : {reg['signal_macd']}")
    print(f"  Volatilité   : {reg['vol_20j_ann']:.2f}% ann. → {reg['signal_vol']}")

    print("\n📉 DRAWDOWN")
    dd = rapport['sp500']['drawdown']
    print(f"  Drawdown actuel  : {dd['drawdown_actuel']*100:.2f}%")
    print(f"  Max global       : {dd['drawdown_max_global']*100:.2f}% (le {dd['date_drawdown_max_global']})")
    print(f"  Max 1 an         : {dd['drawdown_max_1an']*100:.2f}%")

    print("\n📈 PERFORMANCES")
    perf = rapport['sp500']['performances']
    print(f"  5 jours  : {perf['perf_5j']:+.2f}%")
    print(f"  1 mois   : {perf['perf_1m']:+.2f}%")
    print(f"  3 mois   : {perf['perf_3m']:+.2f}%")
    print(f"  1 an     : {perf['perf_1an']:+.2f}%")
    print(f"  YTD      : {perf['perf_ytd']:+.2f}%")
    print(f"  Sharpe 1an : {perf['sharpe_1an_actuel']:.4f}")

    print("\n⚠️  RISQUE (VaR / CVaR)")
    r = rapport['sp500']['risque_queues']
    print(f"  VaR 95%  (1j)  : {r['VaR_95_journalier']:.4f}%")
    print(f"  VaR 99%  (1j)  : {r['VaR_99_journalier']:.4f}%")
    print(f"  CVaR 95% (1j)  : {r['CVaR_95_journalier']:.4f}%")

    print("\n🔗 CORRÉLATION SP500 / VIX")
    corr = rapport['correlation_sp500_vix']
    print(f"  Corrélation 3m  : {corr['correlation_3m']:.4f}")
    print(f"  Corrélation 1an : {corr['correlation_1an']:.4f}")
    print(f"  Signal stress   : {corr['signal_stress']}")

    # ── Sauvegarde JSON
    date_str   = datetime.today().strftime('%Y-%m-%d')
    json_path  = os.path.join(OUTPUT_DIR, f"rapport_{date_str}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ Rapport sauvegardé → {json_path}")

    # ── Sauvegarde SQLite (table rapports)
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rapports_quotidiens (
            date        TEXT PRIMARY KEY,
            rapport_json TEXT,
            inserted_at  TEXT
        )
    """)
    conn.execute("""
        INSERT OR REPLACE INTO rapports_quotidiens
        (date, rapport_json, inserted_at) VALUES (?, ?, ?)
    """, (date_str, json.dumps(rapport, default=str),
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()
    print("✅ Rapport stocké dans SQLite → table 'rapports_quotidiens'")

    return rapport


# ─────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────
if __name__ == "__main__":
    rapport = generer_rapport()
    print("\nAnalyse terminée !")