import yfinance as yf
import pandas as pd
import sqlite3
import os
from datetime import datetime

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
os.makedirs("data", exist_ok=True)
DB_PATH = "data/market_data.db"

actifs = {
    'sp500': '^GSPC',
    'vix'  : '^VIX',
    
}

def get_connection():
    return sqlite3.connect(DB_PATH)

def creer_tables():
    conn = get_connection()
    cursor = conn.cursor()

    for nom in actifs:
        # Table prix bruts
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {nom}_prices (
                date        TEXT PRIMARY KEY,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                inserted_at TEXT
            )
        """)

        # Table features calculées
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {nom}_features (
                date        TEXT PRIMARY KEY,
                close       REAL,
                rendement   REAL,
                vol_20j     REAL,
                vol_5j      REAL,
                ma_20       REAL,
                ma_50       REAL,
                ma_ratio    REAL,
                rsi         REAL,
                macd        REAL,
                macd_signal REAL,
                inserted_at TEXT
            )
        """)

    # Table logs des mises à jour
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS update_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            actif       TEXT,
            lignes_ajoutees INTEGER,
            date_debut  TEXT,
            date_fin    TEXT,
            timestamp   TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Tables SQLite créées.")

def telecharger_et_stocker(nom, ticker):
    print(f"\nTéléchargement de {nom} ({ticker})...")

   # ── DAILY (long terme → modèle principal)
    raw = yf.download(ticker, start="2000-01-01", interval='1d', progress=False)

# ── INTRADAY (optionnel pour futur)
    raw_1h = yf.download(ticker, period='730d', interval='1h', progress=False)
    raw_15min = yf.download(ticker, period='60d', interval='15m', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df.dropna(inplace=True)
    df = df[~df.index.duplicated()]
    df.sort_index(inplace=True)

    # Ajouter timestamp
    df['inserted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Sauvegarder dans SQLite
    conn = get_connection()
    df.index = df.index.strftime('%Y-%m-%d')
    df.columns = ['open','high','low','close','volume','inserted_at']
    df.to_sql(f"{nom}_prices", conn,
              if_exists='replace', index=True, index_label='date')
    conn.close()

    print(f"  {len(df)} lignes → SQLite table '{nom}_prices'")
    print(f"  Période : {df.index[0]} → {df.index[-1]}")
    return df

def calculer_et_stocker_features(nom):
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {nom}_prices",
                     conn, index_col='date', parse_dates=['date'])
    conn.close()

    df.columns = [c.lower() for c in df.columns]
    df['rendement']   = df['close'].pct_change()
    df['vol_20j']     = df['rendement'].rolling(20).std()
    df['vol_5j']      = df['rendement'].rolling(5).std()
    df['ma_20']       = df['close'].rolling(20).mean()
    df['ma_50']       = df['close'].rolling(50).mean()
    df['ma_ratio']    = df['ma_20'] / df['ma_50']

    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    ema12           = df['close'].ewm(span=12).mean()
    ema26           = df['close'].ewm(span=26).mean()
    df['macd']      = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['inserted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    df.dropna(inplace=True)

    feat_cols = ['close','rendement','vol_20j','vol_5j',
                 'ma_20','ma_50','ma_ratio','rsi',
                 'macd','macd_signal','inserted_at']

    conn = get_connection()
    df[feat_cols].to_sql(f"{nom}_features", conn,
                          if_exists='replace',
                          index=True, index_label='date')
    conn.close()
    print(f"  Features → SQLite table '{nom}_features'")

# ─────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("INITIALISATION BASE DE DONNEES SQLITE")
    print("=" * 50)

    creer_tables()

    for nom, ticker in actifs.items():
        telecharger_et_stocker(nom, ticker)
        calculer_et_stocker_features(nom)

    # Vérification
    conn = get_connection()
    print("\nVERIFICATION DES TABLES :")
    for nom in actifs:
        n = pd.read_sql(
            f"SELECT COUNT(*) as n FROM {nom}_prices", conn
        )['n'][0]
        print(f"  {nom}_prices    : {n} lignes")
        n = pd.read_sql(
            f"SELECT COUNT(*) as n FROM {nom}_features", conn
        )['n'][0]
        print(f"  {nom}_features  : {n} lignes")
    conn.close()

    print("\nBase SQLite créée → data/market_data.db")
    print("Initialisation terminée !")