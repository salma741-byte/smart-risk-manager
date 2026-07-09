# ============================================================
# 2_merge_dataset.py — DATASET GLOBAL MULTI-ACTIFS
# ============================================================

import pandas as pd
import sqlite3

DB_PATH = "data/market_data.db"

actifs = ['sp500', 'vix', 'bitcoin', 'gold', 'dxy']

def load_features(actif):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {actif}_features",
                     conn, parse_dates=['date'])
    conn.close()

    df.set_index('date', inplace=True)

    # renommer colonnes
    df = df.add_prefix(f"{actif}_")

    return df

def merge_all():
    dfs = []

    for actif in actifs:
        df = load_features(actif)
        dfs.append(df)

    # merge sur date
    df_final = pd.concat(dfs, axis=1, join='inner')

    print(f"Dataset fusionné : {df_final.shape}")

    return df_final

def add_cross_features(df):

    # différences entre actifs (TRÈS IMPORTANT)
    df['sp500_vs_btc']  = df['sp500_rendement'] - df['bitcoin_rendement']
    df['sp500_vs_gold'] = df['sp500_rendement'] - df['gold_rendement']

    # VIX relatif
    df['vix_spike'] = df['vix_close'] / df['vix_ma_20']

    # momentum global
    df['risk_on'] = df['bitcoin_rendement'] - df['vix_rendement']

    return df

def save_dataset(df):
    conn = sqlite3.connect(DB_PATH)

    df.to_sql("global_dataset",
              conn,
              if_exists='replace',
              index=True)

    conn.close()
    print("Dataset global sauvegardé ✓")

if __name__ == "__main__":
    print("="*50)
    print("CREATION DATASET GLOBAL")
    print("="*50)

    df = merge_all()
    df = add_cross_features(df)

    df.dropna(inplace=True)

    print(df.head())

    save_dataset(df)

    print("\nTerminé ✓")