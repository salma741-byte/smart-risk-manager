import yfinance as yf
import pandas as pd
import sqlite3
import subprocess
import os
from datetime import datetime, timedelta

DB_PATH = "data/market_data.db"

actifs = {
    'sp500'  : '^GSPC',
    'vix'    : '^VIX',
    'bitcoin': 'BTC-USD',
    'gold'   : 'GC=F',
    'dxy'    : 'DX-Y.NYB'
}
def get_connection():
    return sqlite3.connect(DB_PATH)

def mettre_a_jour(nom, ticker):
    conn = get_connection()

    # Dernière date en base
    result = pd.read_sql(
        f"SELECT MAX(date) as last_date FROM {nom}_prices",
        conn
    )
    conn.close()

    derniere_date = pd.Timestamp(result['last_date'][0])
    date_debut    = derniere_date + timedelta(days=1)
    date_fin      = datetime.today()

    print(f"\n{nom.upper()} — dernière date : {derniere_date.date()}")

    if date_debut.date() >= date_fin.date():
        print(f"  Déjà à jour.")
        return 0

    print(f"  Téléchargement du {date_debut.date()}...")

    df_nouveau = yf.download(
        ticker,
        start=date_debut.strftime('%Y-%m-%d'),
        end=date_fin.strftime('%Y-%m-%d'),
        progress=False
    )

    if df_nouveau.empty:
        print(f"  Aucune nouvelle donnée.")
        return 0

    if isinstance(df_nouveau.columns, pd.MultiIndex):
        df_nouveau.columns = df_nouveau.columns.get_level_values(0)

    df_nouveau = df_nouveau[['Open','High','Low','Close','Volume']].copy()
    df_nouveau.index = pd.to_datetime(df_nouveau.index)
    df_nouveau.index = df_nouveau.index.strftime('%Y-%m-%d')
    df_nouveau.index.name = 'date'
    df_nouveau.columns     = ['open','high','low','close','volume']
    df_nouveau['inserted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_nouveau.dropna(inplace=True)

    # Insérer uniquement les nouvelles lignes
    conn = get_connection()
    df_nouveau.to_sql(f"{nom}_prices", conn,
                      if_exists='append', index=True)

    # Supprimer les doublons
    conn.execute(f"""
        DELETE FROM {nom}_prices
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM {nom}_prices GROUP BY date
        )
    """)

    # Logger la mise à jour
    conn.execute("""
        INSERT INTO update_logs
        (actif, lignes_ajoutees, date_debut, date_fin, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (nom, len(df_nouveau),
          str(date_debut.date()),
          str(date_fin.date()),
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    conn.commit()
    conn.close()

    print(f"  {len(df_nouveau)} nouvelles lignes ajoutées.")
    return len(df_nouveau)

def recalculer_features(nom):
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT * FROM {nom}_prices ORDER BY date",
        conn, index_col='date'
    )
    conn.close()

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

    ema12             = df['close'].ewm(span=12).mean()
    ema26             = df['close'].ewm(span=26).mean()
    df['macd']        = ema12 - ema26
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
    print(f"  Features recalculées → {nom}_features")

def push_github(message):
    try:
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', message], check=True)
        subprocess.run(['git', 'push'], check=True)
        print(f"  GitHub : commit pushed — {message}")
    except subprocess.CalledProcessError as e:
        print(f"  GitHub : erreur push — {e}")

# ─────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print(f"Mise à jour du {datetime.today().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 50)

    total_lignes = 0
    for nom, ticker in actifs.items():
        n = mettre_a_jour(nom, ticker)
        recalculer_features(nom)
        total_lignes += n

    # Push automatique si nouvelles données
    if total_lignes > 0:
        message = f"data: update {datetime.today().strftime('%Y-%m-%d')} — {total_lignes} nouvelles lignes"
        push_github(message)
    else:
        print("\nAucune nouvelle donnée — pas de commit.")

    print("\nMise à jour terminée !")