# ============================================================
#  1b_features_extreme.py — CIBLE MOUVEMENTS EXTRÊMES
#  Au lieu de prédire hausse/baisse (trop bruité),
#  on prédit si le S&P va faire un mouvement > seuil
#  dans les 5 prochains jours.
#
#  3 classes :
#    2 = FORTE HAUSSE  (> +1.5%)
#    1 = NEUTRE        (entre -1.5% et +1.5%)
#    0 = FORTE BAISSE  (< -1.5%)
#
#  Stratégie : on ne trade QUE les extrêmes
#  En neutre → cash
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import requests
import warnings
warnings.filterwarnings('ignore')
import os

DB_PATH  = "data/market_data.db"
SEUIL    = 0.015   # 1.5% — seuil mouvement extrême

def calculer_rsi(serie, periode=14):
    delta = serie.diff()
    gain  = delta.clip(lower=0).rolling(periode).mean()
    loss  = (-delta.clip(upper=0)).rolling(periode).mean()
    return 100 - (100 / (1 + gain / loss))

# ─────────────────────────────────────────
# FEATURES DE BASE
# ─────────────────────────────────────────
def features_base(df, vix):
    df = df.join(vix[['vix_close']], how='left')

    # Rendements multi-horizons
    for n in [1, 2, 3, 5, 10, 20]:
        df[f'ret_{n}d'] = df['close'].pct_change(n)

    # Volatilité rolling
    for n in [5, 10, 20, 60]:
        df[f'vol_{n}d'] = df['ret_1d'].rolling(n).std()

    # Moyennes mobiles
    for n in [10, 20, 50, 100, 200]:
        df[f'ma_{n}'] = df['close'].rolling(n).mean()

    df['prix_vs_ma20']  = df['close'] / df['ma_20']  - 1
    df['prix_vs_ma50']  = df['close'] / df['ma_50']  - 1
    df['prix_vs_ma200'] = df['close'] / df['ma_200'] - 1
    df['ma_ratio']      = df['ma_20'] / df['ma_50']
    df['ratio_50_200']  = df['ma_50'] / df['ma_200']

    # RSI multi-périodes
    for n in [2, 7, 14, 21]:
        df[f'rsi_{n}'] = calculer_rsi(df['close'], n)

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # Bollinger
    ma20  = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_zscore'] = (df['close'] - ma20) / std20
    df['bb_width']  = (2 * std20) / ma20   # largeur relative

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_14']  = tr.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # Volume
    df['vol_ratio']  = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_trend']  = df['volume'].pct_change(5)
    df['vol_ratio60']= df['volume'] / df['volume'].rolling(60).mean()

    # Chandeliers
    df['corps']       = (df['close'] - df['open']).abs() / df['open']
    df['ombre_haute'] = (df['high'] - pd.concat([df['close'],df['open']],axis=1).max(axis=1)) / df['open']
    df['ombre_basse'] = (pd.concat([df['close'],df['open']],axis=1).min(axis=1) - df['low']) / df['open']
    df['close_pos']   = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['gap']         = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    return df

# ─────────────────────────────────────────
# FEATURES SPÉCIFIQUES AUX EXTRÊMES
# ─────────────────────────────────────────
def features_extremes(df):
    close = df['close']

    # ── Drawdowns ────────────────────────────────────────────────
    for n in [5, 10, 20, 50, 100]:
        df[f'dd_{n}d'] = (close / close.rolling(n).max() - 1) * 100

    # ── Compression de volatilité (Bollinger Squeeze) ────────────
    # Quand la volatilité est très basse → explosion imminente
    df['bb_squeeze']     = (df['bb_width'] < df['bb_width'].rolling(60).quantile(0.20)).astype(int)
    df['vol_compression']= df['vol_20d'] / df['vol_60d'].replace(0, np.nan)

    # ── Momentum accélération ────────────────────────────────────
    df['mom_accel_3']   = df['ret_1d'].rolling(3).mean()
    df['mom_accel_5']   = df['ret_1d'].rolling(5).mean()
    df['mom_accel_10']  = df['ret_1d'].rolling(10).mean()

    # Est-ce que le momentum s'accélère ?
    df['accel_signal']  = (df['mom_accel_3'] > df['mom_accel_5']).astype(int)

    # ── Jours consécutifs de hausse/baisse ───────────────────────
    ret_pos = (close.pct_change() > 0).astype(int)
    ret_neg = (close.pct_change() < 0).astype(int)

    def consecutive(serie):
        counts = []
        count  = 0
        for v in serie:
            if v == 1:
                count += 1
            else:
                count = 0
            counts.append(count)
        return pd.Series(counts, index=serie.index)

    df['jours_hausse_consec'] = consecutive(ret_pos)
    df['jours_baisse_consec'] = consecutive(ret_neg)

    # ── VIX features avancées ────────────────────────────────────
    if 'vix_close' in df.columns:
        df['vix_ret_1d']       = df['vix_close'].pct_change(1)
        df['vix_ret_5d']       = df['vix_close'].pct_change(5)
        df['vix_ma20']         = df['vix_close'].rolling(20).mean()
        df['vix_ma60']         = df['vix_close'].rolling(60).mean()
        df['vix_spike']        = df['vix_close'] / df['vix_ma20'] - 1
        df['vix_pct_rank_252'] = df['vix_close'].rolling(252).rank(pct=True)
        df['vix_extreme_hi']   = (df['vix_close'] > 30).astype(int)
        df['vix_extreme_lo']   = (df['vix_close'] < 12).astype(int)
        df['vix_retour']       = ((df['vix_close'] < df['vix_close'].shift(1)) &
                                   (df['vix_close'] > 25)).astype(int)
        # VIX/SP500 divergence
        df['sp_vix_div']       = df['ret_1d'] + df['vix_ret_1d']

    # ── Z-scores multi-périodes ──────────────────────────────────
    for p in [20, 60, 252]:
        ma  = close.rolling(p).mean()
        std = close.rolling(p).std()
        df[f'zscore_{p}'] = (close - ma) / std.replace(0, np.nan)

    # ── Cassures de niveaux avec volume ─────────────────────────
    df['breakout_20d']  = ((close > close.rolling(20).max().shift(1)) &
                            (df['vol_ratio'] > 1.5)).astype(int)
    df['breakout_50d']  = ((close > close.rolling(50).max().shift(1)) &
                            (df['vol_ratio'] > 1.5)).astype(int)
    df['breakdown_20d'] = ((close < close.rolling(20).min().shift(1)) &
                            (df['vol_ratio'] > 1.5)).astype(int)

    # ── Skewness des rendements (asymétrie) ─────────────────────
    df['ret_skew_20'] = df['ret_1d'].rolling(20).skew()
    df['ret_kurt_20'] = df['ret_1d'].rolling(20).kurt()

    # ── Cohérence du momentum ────────────────────────────────────
    df['mom_coherent'] = (
        (close > close.shift(1)).astype(int) +
        (close > close.shift(5)).astype(int) +
        (close > close.shift(20)).astype(int)
    )

    # ── Historique des extrêmes passés ──────────────────────────
    df['nb_extreme_hausse_20d'] = (df['ret_1d'] > SEUIL/5).rolling(20).sum()
    df['nb_extreme_baisse_20d'] = (df['ret_1d'] < -SEUIL/5).rolling(20).sum()
    # ── ADX — force de tendance ──────────────────────────────────
    high, low = df['high'], df['low']
    plus_dm   = (high.diff()).clip(lower=0)
    minus_dm  = (-low.diff()).clip(lower=0)
    plus_dm   = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm  = minus_dm.where(minus_dm > plus_dm, 0)
    tr        = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr14     = tr.rolling(14).mean()
    plus_di   = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di  = 100 * minus_dm.rolling(14).mean() / atr14
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['adx_14']   = dx.rolling(14).mean()
    df['di_signal']= (plus_di > minus_di).astype(int)

    # ── Saisonnalité ─────────────────────────────────────────────
    df['day_of_week'] = df.index.dayofweek
    df['month']       = df.index.month
    df['is_january']  = (df.index.month == 1).astype(int)
    df['is_q4']       = df.index.month.isin([10,11,12]).astype(int)

    return df

# ─────────────────────────────────────────
# FEAR & GREED
# ─────────────────────────────────────────
def ajouter_fear_greed(df):
    try:
        url = "https://api.alternative.me/fng/?limit=3000&format=json"
        r   = requests.get(url, timeout=10).json()
        fg  = pd.DataFrame(r['data'])
        fg['date']     = pd.to_datetime(fg['timestamp'].astype(int), unit='s').dt.normalize()
        fg['fg_index'] = fg['value'].astype(float)
        fg  = fg.set_index('date')[['fg_index']].sort_index()
        fg  = fg[~fg.index.duplicated()]
        df  = df.join(fg, how='left')
        df['fg_index']         = df['fg_index'].ffill().fillna(50)
        df['fg_change_5d']     = df['fg_index'].diff(5)
        df['fg_change_20d']    = df['fg_index'].diff(20)
        df['fg_extreme_fear']  = (df['fg_index'] < 20).astype(int)  # peur extrême
        df['fg_extreme_greed'] = (df['fg_index'] > 80).astype(int)  # cupidité extrême
        df['fg_zscore']        = ((df['fg_index'] - df['fg_index'].rolling(30).mean()) /
                                   df['fg_index'].rolling(30).std())
        df['fg_pct_rank']      = df['fg_index'].rolling(252).rank(pct=True)
        print("  ✓ Fear & Greed chargé")
    except Exception as e:
        print(f"  ⚠️  Fear & Greed non dispo ({e})")
        for col in ['fg_index','fg_change_5d','fg_change_20d',
                    'fg_extreme_fear','fg_extreme_greed',
                    'fg_zscore','fg_pct_rank']:
            df[col] = 0
        df['fg_index'] = 50
    return df

# ─────────────────────────────────────────
# CIBLE — MOUVEMENTS EXTRÊMES
# ─────────────────────────────────────────
def creer_cible_extreme(df, seuil=SEUIL, horizon=5):
    """
    3 classes :
      2 = FORTE HAUSSE  (ret_5j > +seuil)
      0 = FORTE BAISSE  (ret_5j < -seuil)
      1 = NEUTRE        (entre -seuil et +seuil)

    Pour le modèle binaire :
      target_up   = 1 si FORTE HAUSSE
      target_down = 1 si FORTE BAISSE
    """
    ret_futur = df['close'].shift(-horizon) / df['close'] - 1

    df['ret_futur_5j']  = ret_futur
    df['classe_extreme']= np.where(ret_futur > seuil,  2,
                          np.where(ret_futur < -seuil, 0, 1))

    # Cible binaire : forte hausse vs reste
    df['target_up']   = (df['classe_extreme'] == 2).astype(int)
    # Cible binaire : forte baisse vs reste
    baisse_regime = (
    (df['close'] < df['ma_50']) &
    (df['ma_50'] < df['ma_200']) &
    (df['rsi_14'] < 45)
).astype(int)
    df['target_down'] = baisse_regime
    n_up   = (df['classe_extreme'] == 2).sum()
    n_neut = (df['classe_extreme'] == 1).sum()
    n_down = (df['classe_extreme'] == 0).sum()
    n_tot  = len(df)
    print(f"\n  Distribution des classes (seuil={seuil*100:.1f}%) :")
    print(f"    FORTE HAUSSE : {n_up}   ({n_up/n_tot*100:.1f}%)")
    print(f"    NEUTRE       : {n_neut} ({n_neut/n_tot*100:.1f}%)")
    print(f"    FORTE BAISSE : {n_down} ({n_down/n_tot*100:.1f}%)")

    return df

# ─────────────────────────────────────────
# FEATURES FINALES PAR RÉGIME
# ─────────────────────────────────────────
FEATURES_EXTREMES = [
    # Volatilité — clé pour les extrêmes
    'vol_5d', 'vol_10d', 'vol_20d', 'vol_60d',
    'atr_pct', 'bb_width', 'bb_squeeze', 'vol_compression',
    'bb_zscore', 'zscore_20', 'zscore_60',

    # Drawdowns — position dans la chute
    'dd_5d', 'dd_10d', 'dd_20d', 'dd_50d',

    # Momentum
    'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d',
    'mom_accel_3', 'mom_accel_5', 'mom_coherent',
    'jours_hausse_consec', 'jours_baisse_consec',
    'accel_signal',

    # RSI — survente/surachat
    'rsi_2', 'rsi_7', 'rsi_14', 'rsi_21',

    # MACD
    'macd_hist',

    # VIX — peur du marché
    'vix_close', 'vix_ret_1d', 'vix_spike',
    'vix_pct_rank_252', 'vix_extreme_hi', 'vix_extreme_lo',
    'vix_retour', 'sp_vix_div',

    # Sentiment
    'fg_index', 'fg_change_5d', 'fg_change_20d',
    'fg_extreme_fear', 'fg_extreme_greed', 'fg_pct_rank',

    # Cassures avec volume
    'breakout_20d', 'breakout_50d', 'breakdown_20d',
    'vol_ratio', 'vol_ratio60',

    # Chandeliers
    'corps', 'ombre_basse', 'close_pos', 'gap',

    # Structure de marché
    'adx_14', 'di_signal',
    'prix_vs_ma20', 'prix_vs_ma50', 'prix_vs_ma200',
    'ret_skew_20', 'ret_kurt_20',
    'nb_extreme_hausse_20d', 'nb_extreme_baisse_20d',

    # Saisonnalité
    'day_of_week', 'month', 'is_january', 'is_q4',
]

# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def construire_dataset_extreme():
    conn = sqlite3.connect(DB_PATH)
    sp   = pd.read_sql("SELECT * FROM sp500_prices", conn,
                        index_col='date', parse_dates=['date'])
    vix  = pd.read_sql("SELECT date, close AS vix_close FROM vix_prices",
                        conn, index_col='date', parse_dates=['date'])
    conn.close()

    sp.columns = [c.lower() for c in sp.columns]
    df = sp[['open','high','low','close','volume']].copy()
    df = df[df.index.weekday < 5]
    df.sort_index(inplace=True)

    print("Calcul des features de base...")
    df = features_base(df, vix)

    print("Calcul des features spécifiques aux extrêmes...")
    df = features_extremes(df)

    print("Ajout Fear & Greed...")
    df = ajouter_fear_greed(df)

    print("Création de la cible...")
    df = creer_cible_extreme(df, seuil=SEUIL, horizon=5)

    # Nettoyage
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[df.index.weekday < 5]

    print(f"\n  Dataset final : {len(df)} lignes × {df.shape[1]} colonnes")
    print(f"  Période : {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

    # Sauvegarde SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sp500_extreme_features", conn,
              if_exists='replace', index=True, index_label='date')
    pd.DataFrame({'feature': FEATURES_EXTREMES}).to_sql(
        'features_extremes', conn, if_exists='replace', index=False)
    conn.close()

    print("  → Sauvegardé dans sp500_extreme_features ✓")
    return df


if __name__ == "__main__":
    print("=" * 55)
    print("  FEATURES MOUVEMENTS EXTRÊMES S&P500")
    print("=" * 55)
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)
    df = construire_dataset_extreme()
    print("\nRelance : python 2_train_extreme.py")