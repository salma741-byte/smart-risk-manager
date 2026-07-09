# ============================================================
#  3_allocation_dynamique.py
#  Système d'exposition progressive (0.0 → 1.0)
#  Remplace le signal binaire ACHAT/VENTE/NEUTRE
#
#  Logique :
#    Régime CALME  → BUY par défaut, ML ajuste l'exposition
#    Régime STRESS → ML décide l'exposition, protection prioritaire
#
#  Exposition = fraction du capital investi (0=cash, 1=fully invested)
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
import joblib
import os
from datetime import datetime

os.makedirs("results", exist_ok=True)
DB_PATH = "data/market_data.db"


# ─────────────────────────────────────────────────────────────
# 1. CALCUL DE L'EXPOSITION PAR RÉGIME
# ─────────────────────────────────────────────────────────────

# Paliers d'exposition autorisés (évite les valeurs continues difficiles à exécuter)
PALIERS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

def _arrondir_palier(valeur):
    """Arrondit une exposition continue au palier le plus proche."""
    return min(PALIERS, key=lambda x: abs(x - valeur))


def proba_vers_exposition(proba, regime, vix=None):
    """
    Convertit P(hausse) en exposition (0.0 → 1.0).

    RÉGIME CALME  (VIX < 20) — marché tendanciel haussier
    ────────────────────────────────────────────────────────
    Principe : BUY par défaut, le modèle module à la baisse.
    Le marché monte 61% du temps en régime calme (stats confirmées).
    On ne sort JAMAIS totalement sauf signal très fort baissier.

      P >= 0.65 → 1.0   position pleine  (forte conviction haussière)
      P >= 0.60 → 0.8   position normale
      P >= 0.55 → 0.6   position de base (default calme)
      P >= 0.50 → 0.4   légèrement réduit
      P >= 0.44 → 0.3   prudence, réduction
      P <  0.44 → 0.0   seul vrai signal de sortie (rare)

    RÉGIME STRESS (VIX 20-30) — marché imprévisible
    ────────────────────────────────────────────────────────
    Principe : le modèle décide, mais avec un plancher à 0.3
    si le VIX commence à baisser (signal de fin de stress).
    On évite le short sauf conviction forte baissière.

      P >= 0.58 → 0.8   rebond probable, position significative
      P >= 0.52 → 0.5   conviction modérée
      P >= 0.45 → 0.3   incertain, exposition minimale
      P >= 0.38 → 0.0   cash, pas de conviction
      P <  0.38 → -0.2  short léger (conviction baissière forte)

    RÉGIME CRASH  (VIX > 30) — marché en panique
    ────────────────────────────────────────────────────────
    Principe : protection maximale, ne pas attraper le couteau.
    On attend un signal de stabilisation clair avant de rentrer.

      P >= 0.65 → 0.5   rebond de panique possible (capitulation)
      P >= 0.55 → 0.2   timide, juste un orteil
      P <  0.55 → 0.0   cash total
      P <  0.40 → -0.2  couverture légère

    Override VIX
    ────────────────────────────────────────────────────────
    Si VIX fourni et commence à baisser depuis un pic :
    → plancher d'exposition relevé (signal de détente du marché)
    """
    if regime == 'calme':
        if   proba >= 0.65: expo = 1.0
        elif proba >= 0.60: expo = 0.8
        elif proba >= 0.55: expo = 0.6
        elif proba >= 0.50: expo = 0.4
        elif proba >= 0.44: expo = 0.3
        else:               expo = 0.0

    elif regime == 'stress':
        if   proba >= 0.58: expo = 0.8
        elif proba >= 0.52: expo = 0.5
        elif proba >= 0.45: expo = 0.3
        elif proba >= 0.38: expo = 0.0
        else:               expo = -0.2

        # Override : si VIX < 25 (sortie de stress), plancher à 0.3
        if vix is not None and vix < 25:
            expo = max(expo, 0.3)

    else:  # crash (VIX > 30)
        if   proba >= 0.65: expo = 0.5
        elif proba >= 0.55: expo = 0.2
        elif proba >= 0.40: expo = 0.0
        else:               expo = -0.2

        # Override : si VIX commence à baisser (VIX 30-35 et en baisse)
        if vix is not None and 28 < vix < 35:
            expo = max(expo, 0.2)

    return expo


def signal_texte(exposition):
    """Traduit une exposition en label lisible."""
    if   exposition >= 0.8:  return "FORT ACHAT"
    elif exposition >= 0.5:  return "ACHAT"
    elif exposition >= 0.3:  return "REDUIT"
    elif exposition == 0.0:  return "NEUTRE / CASH"
    elif exposition > -0.3:  return "SHORT LEGER"
    else:                    return "SHORT"


def _lisser_exposition(series_expo, alpha_ewm=0.3):
    """
    Lissage exponentiel de l'exposition + arrondi aux paliers.

    Évite les changements de position trop fréquents
    tout en réagissant aux vrais changements de signal.

    alpha_ewm = 0.3 → réactivité modérée
      (0.1 = très lent, 0.5 = réactif)
    """
    expo_lisse = []
    prev_smooth = series_expo.iloc[0]

    for i, curr in enumerate(series_expo):
        # Lissage exponentiel
        smoothed = alpha_ewm * curr + (1 - alpha_ewm) * prev_smooth
        # Arrondi au palier le plus proche
        arrondi  = _arrondir_palier(smoothed)
        expo_lisse.append(arrondi)
        prev_smooth = smoothed   # on lisse sur la valeur continue, pas arrondie

    return expo_lisse


# ─────────────────────────────────────────────────────────────
# 2. BACKTEST AVEC EXPOSITION DYNAMIQUE
# ─────────────────────────────────────────────────────────────

def backtest_allocation(df_test, probas, regime_label,
                        cout_transaction=0.001):
    """
    Simule la stratégie avec exposition dynamique.
    Prend en compte les coûts de transaction à chaque changement.

    Paramètres
    ----------
    probas           : array P(hausse) pour chaque jour
    cout_transaction : 0.1% par trade (réaliste pour ETF/futures)
    """
    df = df_test.copy().reset_index()

    # Déterminer le sous-régime (calme / stress / crash)
    if regime_label == 'calme':
        regime = 'calme'
    elif 'vix_close' in df.columns:
        # Stress vs crash selon VIX
        regime = None   # calculé jour par jour via vix_close
    else:
        regime = 'stress'

    # Calcul exposition jour par jour (avec VIX si disponible)
    df['proba'] = probas
    expos = []
    for i, row in df.iterrows():
        vix_val = row['vix_close'] if 'vix_close' in df.columns else None
        if regime is None:
            # Régime dynamique selon le VIX du jour
            r = ('crash'  if vix_val and vix_val >= 30 else
                 'stress' if vix_val and vix_val >= 20 else 'calme')
        else:
            r = regime
        expos.append(proba_vers_exposition(row['proba'], r, vix=vix_val))

    df['exposition'] = expos

    # Lissage exponentiel + arrondi aux paliers
    df['exposition_lisse'] = _lisser_exposition(df['exposition'], alpha_ewm=0.3)

    # Rendement journalier du sous-jacent
    df['ret_j1'] = df['close'].pct_change(1).shift(-1)
    df.dropna(subset=['ret_j1'], inplace=True)

    # Coût de transaction à chaque changement d'exposition
    df['changement_expo'] = df['exposition_lisse'].diff().abs()
    df['cout']            = df['changement_expo'] * cout_transaction

    # Rendement de la stratégie
    df['ret_strat'] = df['exposition_lisse'] * df['ret_j1'] - df['cout']
    df['ret_bh']    = df['ret_j1']

    # Cumuls
    df['cumul_strat'] = (1 + df['ret_strat']).cumprod()
    df['cumul_bh']    = (1 + df['ret_bh']).cumprod()

    # ── Métriques ────────────────────────────────────────────────
    ret_tot  = df['cumul_strat'].iloc[-1] - 1
    ret_bh   = df['cumul_bh'].iloc[-1] - 1
    alpha    = ret_tot - ret_bh

    std      = df['ret_strat'].std()
    sharpe   = (df['ret_strat'].mean() / std * np.sqrt(252)
                if std > 0 else 0)

    roll_max = df['cumul_strat'].cummax()
    max_dd   = ((df['cumul_strat'] - roll_max) / roll_max).min()

    # Jours investis (exposition > 0)
    jours_investis = (df['exposition_lisse'] > 0).mean() * 100
    expo_moyenne   = df['exposition_lisse'].mean()

    # Hit rate pondéré (jours où on était investi et le marché a fait ce qu'on voulait)
    mask_long  = df['exposition_lisse'] > 0
    mask_short = df['exposition_lisse'] < 0
    hit_long   = (df.loc[mask_long,  'ret_j1'] > 0).mean() if mask_long.sum() > 0 else 0
    hit_short  = (df.loc[mask_short, 'ret_j1'] < 0).mean() if mask_short.sum() > 0 else 0

    print(f"\n  BACKTEST ALLOCATION DYNAMIQUE — {regime_label.upper()}")
    print(f"  {'─'*48}")
    print(f"    Période           : {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"    Nb jours          : {len(df)}")
    print(f"    Exposition moyenne : {expo_moyenne:.2f}  ({jours_investis:.0f}% du temps investi)")
    print(f"    {'─'*48}")
    print(f"    Rendement ML      : {ret_tot*100:+.1f}%")
    print(f"    Rendement B&H     : {ret_bh*100:+.1f}%")
    print(f"    Alpha             : {alpha*100:+.1f}%")
    print(f"    Sharpe            : {sharpe:.2f}")
    print(f"    Max Drawdown      : {max_dd*100:.1f}%")
    print(f"    Hit rate LONG     : {hit_long*100:.1f}%")
    if mask_short.sum() > 0:
        print(f"    Hit rate SHORT    : {hit_short*100:.1f}%")

    # Distribution des expositions
    print(f"\n    Distribution des expositions :")
    for expo_val in sorted(df['exposition_lisse'].unique()):
        pct = (df['exposition_lisse'] == expo_val).mean() * 100
        bar = '█' * int(pct / 2)
        sig = signal_texte(expo_val)
        print(f"      {expo_val:+.1f} ({sig:<16}) {bar} {pct:.1f}%")

    return df, {
        'regime'         : regime_label,
        'ret_strat'      : ret_tot,
        'ret_bh'         : ret_bh,
        'alpha'          : alpha,
        'sharpe'         : sharpe,
        'max_drawdown'   : max_dd,
        'expo_moyenne'   : expo_moyenne,
        'jours_investis' : jours_investis,
    }


# ─────────────────────────────────────────────────────────────
# 3. SIGNAL DE PRODUCTION AVEC EXPOSITION
# ─────────────────────────────────────────────────────────────

def signal_production(df, verbose=True):
    """
    Génère le signal d'allocation pour aujourd'hui.
    """
    derniere = df.iloc[[-1]]
    vix      = derniere['vix_close'].values[0] if 'vix_close' in df.columns else 20
    regime   = 'calme' if vix < 20 else 'stress'

    try:
        modeles  = joblib.load(f"models/ensemble_{regime}.pkl")
        scaler   = joblib.load(f"models/scaler_{regime}.pkl")
        features = joblib.load(f"models/features_{regime}.pkl")
    except FileNotFoundError:
        print(f"  Modèle '{regime}' non trouvé.")
        return None

    feats_dispo = [f for f in features if f in derniere.columns]
    X           = derniere[feats_dispo].values
    X_scaled    = scaler.transform(X)

    # Probas de chaque modèle
    probas_list = []
    poids       = {'rf': 0.30, 'xgb': 0.50, 'lr': 0.20}
    for nom, m in modeles.items():
        p = m['model'].predict_proba(X_scaled)[0, 1]
        w = poids.get(nom, 0.33)
        probas_list.append(p * w)

    proba_finale = sum(probas_list)
    exposition   = proba_vers_exposition(proba_finale, regime)
    signal       = signal_texte(exposition)

    if verbose:
        print("\n" + "═"*50)
        print("  SIGNAL D'ALLOCATION — PRODUCTION")
        print("═"*50)
        print(f"  Date              : {df.index[-1].date()}")
        print(f"  VIX               : {vix:.1f}")
        print(f"  Régime            : {regime.upper()}")
        print(f"  P(hausse 5j)      : {proba_finale*100:.1f}%")
        print(f"  Exposition cible  : {exposition*100:.0f}% du capital")
        print(f"  Signal            : {signal}")
        print("═"*50)

        # Interprétation
        if exposition >= 0.8:
            print("  → Fort signal haussier. Position pleine recommandée.")
        elif exposition >= 0.5:
            print("  → Signal modéré. Position partielle.")
        elif exposition >= 0.3:
            print("  → Marché incertain. Exposition réduite.")
        elif exposition == 0.0:
            print("  → Pas de conviction. Rester en cash.")
        else:
            print("  → Signal baissier. Couverture légère possible.")

    return {
        'date'      : df.index[-1],
        'regime'    : regime,
        'vix'       : vix,
        'proba'     : proba_finale,
        'exposition': exposition,
        'signal'    : signal,
    }


# ─────────────────────────────────────────────────────────────
# 4. COMPARAISON STRATÉGIES
# ─────────────────────────────────────────────────────────────

def comparer_strategies(df_test, probas, regime_label):
    """
    Compare 3 stratégies sur la même période :
      1. Buy & Hold pur
      2. Signal binaire (ancien système)
      3. Allocation dynamique (nouveau système)
    """
    df = df_test.copy().reset_index()
    regime = 'calme' if regime_label == 'calme' else 'stress'

    df['ret_j1']  = df['close'].pct_change(1).shift(-1)
    df['proba']   = probas
    df.dropna(subset=['ret_j1'], inplace=True)

    # Stratégie 1 : Buy & Hold
    df['ret_bh'] = df['ret_j1']

    # Stratégie 2 : Signal binaire (seuils 0.60 / 0.40)
    df['signal_bin'] = np.where(df['proba'] >= 0.60,  1,
                       np.where(df['proba'] <= 0.40, -1, 0))
    df['ret_bin'] = df['signal_bin'] * df['ret_j1']

    # Stratégie 3 : Allocation dynamique
    df['exposition'] = df['proba'].apply(
        lambda p: proba_vers_exposition(p, regime))
    df['ret_dyn'] = df['exposition'] * df['ret_j1']

    # Cumuls
    df['cumul_bh']  = (1 + df['ret_bh']).cumprod()
    df['cumul_bin'] = (1 + df['ret_bin']).cumprod()
    df['cumul_dyn'] = (1 + df['ret_dyn']).cumprod()

    def metriques(col_ret, cumul):
        ret    = cumul.iloc[-1] - 1
        std    = df[col_ret].std()
        sharpe = df[col_ret].mean() / std * np.sqrt(252) if std > 0 else 0
        rm     = cumul.cummax()
        maxdd  = ((cumul - rm) / rm).min()
        return ret, sharpe, maxdd

    r_bh,  s_bh,  d_bh  = metriques('ret_bh',  df['cumul_bh'])
    r_bin, s_bin, d_bin  = metriques('ret_bin', df['cumul_bin'])
    r_dyn, s_dyn, d_dyn  = metriques('ret_dyn', df['cumul_dyn'])

    print(f"\n  COMPARAISON STRATÉGIES — {regime_label.upper()}")
    print(f"  {'─'*55}")
    print(f"  {'Stratégie':<25} {'Rendement':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"  {'─'*55}")
    print(f"  {'Buy & Hold':<25} {r_bh*100:>+9.1f}% {s_bh:>8.2f} {d_bh*100:>7.1f}%")
    print(f"  {'Signal binaire':<25} {r_bin*100:>+9.1f}% {s_bin:>8.2f} {d_bin*100:>7.1f}%")
    print(f"  {'Allocation dynamique':<25} {r_dyn*100:>+9.1f}% {s_dyn:>8.2f} {d_dyn*100:>7.1f}%")
    print(f"  {'─'*55}")

    # Sauvegarde
    df[['date','proba','signal_bin','exposition',
        'ret_bh','ret_bin','ret_dyn',
        'cumul_bh','cumul_bin','cumul_dyn']].to_csv(
        f"results/comparaison_{regime_label}.csv", index=False)
    print(f"  Sauvegardé → results/comparaison_{regime_label}.csv")

    return df


# ─────────────────────────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  ALLOCATION DYNAMIQUE — S&P500 ML")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 55)

    # Charger le dataset
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM sp500_extreme_features",
                        conn, index_col='date', parse_dates=['date'])
    conn.close()
    df.sort_index(inplace=True)

    # Enrichissement minimal (mêmes étapes que 2_train_models_FINAL)
    if 'vix_close' in df.columns:
        df['delta_vix']           = df['vix_close'].diff(1)
        df['vix_pct_rank']        = df['vix_close'].rolling(252, min_periods=60).rank(pct=True)
        df['rolling_corr_sp_vix'] = df['ret_1d'].rolling(20).corr(df['vix_ret_1d'])
        df['vix_mean_reversion']  = df['vix_close'] / df['vix_close'].rolling(252, min_periods=60).mean() - 1

    df['zscore_price_60d'] = (
        (df['close'] - df['close'].rolling(60).mean()) /
        df['close'].rolling(60).std()
    )
    if 'drawdown_20d' not in df.columns:
        df['drawdown_20d'] = (df['close'] / df['close'].rolling(20).max() - 1) * 100
    if 'drawdown_50d' not in df.columns:
        df['drawdown_50d'] = (df['close'] / df['close'].rolling(50).max() - 1) * 100

    df['regime_bin'] = np.where(
        df['vix_close'] < 20 if 'vix_close' in df.columns else True,
        'calme', 'stress')

    df['return_5j'] = df['close'].pct_change(5).shift(-5)
    df['target']    = np.nan
    df.loc[df['return_5j'] >  0.010, 'target'] = 1
    df.loc[df['return_5j'] < -0.010, 'target'] = 0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(subset=['target'], inplace=True)
    df.dropna(inplace=True)

    # Split temporel
    split_idx = int(len(df) * 0.80)
    df_test   = df.iloc[split_idx:]

    print(f"\n  Test set : {df_test.index[0].date()} → {df_test.index[-1].date()}")
    print(f"  Nb jours : {len(df_test)}")

    resultats = []

    for regime in ['calme', 'stress']:
        print(f"\n{'═'*55}")
        print(f"  RÉGIME : {regime.upper()}")
        print(f"{'═'*55}")

        mask = df_test['regime_bin'] == regime
        df_r = df_test[mask]

        if len(df_r) < 10:
            print(f"  ⚠  Pas assez de données — skip")
            continue

        try:
            modeles  = joblib.load(f"models/ensemble_{regime}.pkl")
            scaler   = joblib.load(f"models/scaler_{regime}.pkl")
            features = joblib.load(f"models/features_{regime}.pkl")
        except FileNotFoundError:
            print(f"  ⚠  Modèle '{regime}' non trouvé — lance 2_train_models_FINAL.py")
            continue

        feats_dispo = [f for f in features if f in df_r.columns]
        X           = df_r[feats_dispo]
        X_scaled    = scaler.transform(X)

        # Probas ensemble
        poids  = {'rf': 0.30, 'xgb': 0.50, 'lr': 0.20}
        probas = np.zeros(len(X_scaled))
        for nom, m in modeles.items():
            p = m['model'].predict_proba(X_scaled)[:, 1]
            w = poids.get(nom, 0.33)
            probas += p * w

        # Backtest allocation dynamique
        df_bt, res = backtest_allocation(df_r, probas, regime)
        resultats.append(res)

        # Comparaison des 3 stratégies
        comparer_strategies(df_r, probas, regime)

    # ── Résumé final ─────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  RÉSUMÉ FINAL")
    print("═"*55)
    for r in resultats:
        print(f"\n  {r['regime'].upper()}")
        print(f"    Allocation dyn  : {r['ret_strat']*100:+.1f}%  "
              f"Sharpe={r['sharpe']:.2f}  MaxDD={r['max_drawdown']*100:.1f}%")
        print(f"    Buy & Hold      : {r['ret_bh']*100:+.1f}%")
        print(f"    Alpha           : {r['alpha']*100:+.1f}%")
        print(f"    Expo moyenne    : {r['expo_moyenne']:.2f}  "
              f"({r['jours_investis']:.0f}% du temps investi)")

    # ── Signal du jour ───────────────────────────────────────────
    print("\n" + "═"*55)
    signal_production(df)

    print("\nTerminé ✓")
    print("Lance : python 3_allocation_dynamique.py")


