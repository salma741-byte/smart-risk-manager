import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

actifs = {
    'sp500': '^GSPC',
    'vix'  : '^VIX',
    'btc'  : 'BTC-USD'
}

def telecharger_historique(nom, ticker):
    print(f"Téléchargement de {nom} ({ticker})...")

    raw = yf.download(ticker, start="2018-01-01", progress=False)

    # Aplatir les colonnes si MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df.dropna(inplace=True)
    df = df[~df.index.duplicated()]
    df.sort_index(inplace=True)

    chemin = f"data/{nom}.csv"
    df.to_csv(chemin)

    print(f"  {len(df)} lignes sauvegardées → {chemin}")
    print(f"  Période : {df.index[0].date()} → {df.index[-1].date()}")

for nom, ticker in actifs.items():
    telecharger_historique(nom, ticker)

print("\nInitialisation terminée !")