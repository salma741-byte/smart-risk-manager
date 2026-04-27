# ============================================================
#  1c_support_resistance.py
#  Détection automatique supports/résistances + signals SR
#  À lancer APRÈS 1b, AVANT 2_train_models
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "data/market_data.db"

# ─────────────────────────────────────────
# 1. DÉTECTION DES PIVOTS (hauts/bas locaux)
# ─────────────────────────────────────────
def detecter_pivots(df, fenetre=10):
    """
    Détecte les hauts et bas locaux sur une fenêtre glissante.
    Un pivot haut = max local sur 'fenetre' jours de chaque côté
    Un pivot bas  = min local sur 'fenetre' jours de chaque côté
    """
    highs = df['high']
    lows  = df['low']
    close = df['close']
    vol   = df['volume']

    pivot_hauts = []
    pivot_bas   = []

    for i in range(fenetre, len(df) - fenetre):
        # Pivot haut
        if highs.iloc[i] == highs.iloc[i-fenetre:i+fenetre+1].max():
            pivot_hauts.append({
                'date':   df.index[i],
                'prix':   highs.iloc[i],
                'volume': vol.iloc[i],
                'idx':    i
            })
        # Pivot bas
        if lows.iloc[i] == lows.iloc[i-fenetre:i+fenetre+1].min():
            pivot_bas.append({
                'date':   df.index[i],
                'prix':   lows.iloc[i],
                'volume': vol.iloc[i],
                'idx':    i
            })

    return pd.DataFrame(pivot_hauts), pd.DataFrame(pivot_bas)

# ─────────────────────────────────────────
# 2. CLUSTERING DES NIVEAUX (zones SR)
# ─────────────────────────────────────────
def grouper_niveaux(pivots, tolerance_pct=0.02):
    """
    Regroupe les pivots proches en zones SR.
    tolerance_pct = 2% d'écart max pour considérer même zone
    Retourne une liste de zones avec :
    - prix moyen, volume total, fréquence, âge
    """
    if len(pivots) == 0:
        return []

    pivots = pivots.sort_values('prix').reset_index(drop=True)
    zones  = []
    groupe = [pivots.iloc[0]]

    for i in range(1, len(pivots)):
        p_actuel   = pivots.iloc[i]['prix']
        p_groupe   = np.mean([g['prix'] for g in groupe])
        if abs(p_actuel - p_groupe) / p_groupe <= tolerance_pct:
            groupe.append(pivots.iloc[i])
        else:
            zones.append(groupe)
            groupe = [pivots.iloc[i]]
    zones.append(groupe)

    zones_resumees = []
    for z in zones:
        prix_moy  = np.mean([g['prix'] for g in z])
        vol_total = sum([g['volume'] for g in z])
        freq      = len(z)
        date_max  = max([g['date'] for g in z])
        zones_resumees.append({
            'prix':       prix_moy,
            'volume':     vol_total,
            'frequence':  freq,
            'derniere_date': date_max,
        })

    return zones_resumees

# ─────────────────────────────────────────
# 3. SCORE DE QUALITÉ D'UN NIVEAU SR
# ─────────────────────────────────────────
def scorer_niveau(zone, vol_moyen, date_actuelle, vol_moy_global):
    """
    Score 0-100 basé sur les 4 critères du livre :
    - Volume (plus c'est échangé, plus c'est fort)
    - Fréquence (combien de fois le prix est revenu)
    - Âge (plus récent = plus fort)
    - Niveau rond (100, 200, 500... = résistance psychologique)
    """
    # Volume : ratio vs moyenne globale (max 40 pts)
    score_vol  = min(40, (zone['volume'] / max(vol_moy_global, 1)) * 15)

    # Fréquence (max 30 pts)
    score_freq = min(30, zone['frequence'] * 10)

    # Âge : pénalité si vieux (max 20 pts)
    jours_age  = (date_actuelle - zone['derniere_date']).days
    score_age  = max(0, 20 - jours_age / 30)

    # Niveau rond (max 10 pts)
    prix = zone['prix']
    score_rond = 0
    if prix % 100 < 2 or prix % 100 > 98:
        score_rond = 10
    elif prix % 50 < 1.5 or prix % 50 > 48.5:
        score_rond = 6
    elif prix % 10 < 0.5 or prix % 10 > 9.5:
        score_rond = 3

    return round(score_vol + score_freq + score_age + score_rond, 1)

# ─────────────────────────────────────────
# 4. FEATURES SR POUR CHAQUE JOUR
# ─────────────────────────────────────────
def calculer_features_sr(df, lookback=252):
    """
    Pour chaque jour, calcule les features SR en utilisant
    UNIQUEMENT les données passées (pas de data leakage).
    
    Features générées :
    - dist_resistance     : % distance au prochain niveau de résistance
    - dist_support        : % distance au prochain niveau de support
    - score_resistance    : qualité de la résistance (0-100)
    - score_support       : qualité du support (0-100)
    - vol_franchissement  : volume actuel / volume moyen au niveau SR
    - signal_breakout_sr  : 1 si franchissement résistance avec bon volume
    - signal_breakdown_sr : 1 si cassure support avec bon volume
    - zone_compression    : écart support/résistance < 3% (range étroit)
    - proche_resistance   : prix à moins de 2% d'une résistance
    - proche_support      : prix à moins de 2% d'un support
    """
    features = []
    vol_moyen_global = df['volume'].mean()

    print(f"  Calcul SR sur {len(df)} jours (lookback={lookback}j)...")

    for i in range(lookback, len(df)):
        # Données passées uniquement
        hist     = df.iloc[i-lookback:i]
        row      = df.iloc[i]
        close    = row['close']
        vol_jour = row['volume']
        date     = df.index[i]

        # Détecter pivots sur l'historique
        ph, pb = detecter_pivots(hist, fenetre=8)

        # Grouper en zones
        resistances = grouper_niveaux(ph[ph['prix'] > close] if len(ph) > 0 else pd.DataFrame(),
                                       tolerance_pct=0.025)
        supports    = grouper_niveaux(pb[pb['prix'] < close] if len(pb) > 0 else pd.DataFrame(),
                                       tolerance_pct=0.025)

        # Scorer et trier
        for z in resistances:
            z['score'] = scorer_niveau(z, vol_jour, date, vol_moyen_global)
        for z in supports:
            z['score'] = scorer_niveau(z, vol_jour, date, vol_moyen_global)

        resistances = sorted(resistances, key=lambda x: x['prix'])   # plus proche = premier
        supports    = sorted(supports,    key=lambda x: -x['prix'])   # plus proche = premier

        feat = {}

        # ── Distance aux niveaux ──────────────────────────────
        if resistances:
            r1 = resistances[0]
            feat['dist_resistance']  = (r1['prix'] - close) / close * 100
            feat['score_resistance'] = r1['score']
            feat['prix_resistance']  = r1['prix']
        else:
            feat['dist_resistance']  = 10.0
            feat['score_resistance'] = 0.0
            feat['prix_resistance']  = close * 1.10

        if supports:
            s1 = supports[0]
            feat['dist_support']  = (close - s1['prix']) / close * 100
            feat['score_support'] = s1['score']
            feat['prix_support']  = s1['prix']
        else:
            feat['dist_support']  = 10.0
            feat['score_support'] = 0.0
            feat['prix_support']  = close * 0.90

        # ── Signaux de franchissement ─────────────────────────
        vol_moyen_20 = hist['volume'].tail(20).mean()
        vol_ratio    = vol_jour / max(vol_moyen_20, 1)

        # Breakout résistance : prix au-dessus de 3% + volume 2x
        feat['signal_breakout_sr']  = int(
            feat['dist_resistance'] < 0 and   # prix a franchi la résistance
            abs(feat['dist_resistance']) > 0.5 and
            vol_ratio >= 2.0 and
            feat['score_resistance'] >= 30
        )

        # Breakdown support : prix en dessous de 3% + volume 2x
        feat['signal_breakdown_sr'] = int(
            feat['dist_support'] < 0 and      # prix a cassé le support
            abs(feat['dist_support']) > 0.5 and
            vol_ratio >= 2.0 and
            feat['score_support'] >= 30
        )

        # ── Proximité aux niveaux ─────────────────────────────
        feat['proche_resistance'] = int(0 < feat['dist_resistance'] < 2.0)
        feat['proche_support']    = int(0 < feat['dist_support']    < 2.0)

        # ── Zone de compression (range étroit) ───────────────
        ecart_sr = feat['dist_resistance'] + feat['dist_support']
        feat['zone_compression'] = int(ecart_sr < 4.0)

        # ── Ratio volume au niveau ────────────────────────────
        feat['vol_franchissement'] = round(vol_ratio, 3)

        # ── Nb niveaux SR dans la zone ────────────────────────
        feat['nb_resistances'] = len(resistances)
        feat['nb_supports']    = len(supports)

        features.append({'date': date, **feat})

    return pd.DataFrame(features).set_index('date')


# ─────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def ajouter_features_sr():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM sp500_ml_features",
                        conn, index_col='date', parse_dates=['date'])
    conn.close()

    df.sort_index(inplace=True)

    print("Calcul des features Support/Résistance...")
    sr_features = calculer_features_sr(df, lookback=252)

    # Joindre au dataset principal
    df = df.join(sr_features, how='left')

    # Remplir les NaN du début (lookback)
    cols_sr = sr_features.columns.tolist()
    df[cols_sr] = df[cols_sr].ffill().fillna(0)

    # Sauvegarder
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sp500_ml_features", conn,
              if_exists='replace', index=True, index_label='date')
    conn.close()

    print(f"  ✓ {len(cols_sr)} features SR ajoutées : {cols_sr}")
    print(f"  ✓ Dataset mis à jour : {len(df)} lignes × {df.shape[1]} colonnes")

    # Stats rapides
    print(f"\n  Breakouts détectés : {int(df['signal_breakout_sr'].sum())}")
    print(f"  Breakdowns détectés: {int(df['signal_breakdown_sr'].sum())}")
    print(f"  Jours près résistance : {int(df['proche_resistance'].sum())}")
    print(f"  Jours près support    : {int(df['proche_support'].sum())}")

    return df


if __name__ == "__main__":
    print("=" * 55)
    print("  SUPPORT / RÉSISTANCE + VOLUME — FEATURES")
    print("=" * 55)
    df = ajouter_features_sr()
    print("\nRelance : python 2_train_models_FINAL_V3.py")