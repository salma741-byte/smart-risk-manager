import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

actifs = {
    'sp500': '^GSPC',
    'vix'  : '^VIX',
    'btc'  : 'BTC-USD'
}

def mettre_a_jour(nom, ticker):
    chemin = f"data/{nom}.csv"

    if not os.path.exists(chemin):
        print(f"Fichier {chemin} introuvable.")
        return

    df_existant = pd.read_csv(chemin, index_col=0)
    df_existant.index = pd.to_datetime(df_existant.index, format='mixed')
    df_existant.index.name = 'Date'

    derniere_date = pd.Timestamp(df_existant.index[-1])
    date_debut    = derniere_date + timedelta(days=1)
    date_fin      = datetime.today()

    print(f"\n{nom.upper()} — dernière date connue : {derniere_date.date()}")

    if date_debut.date() >= date_fin.date():
        print(f"  Déjà à jour.")
        return df_existant

    print(f"  Téléchargement du {date_debut.date()} au {date_fin.date()}...")

    df_nouveau = yf.download(
        ticker,
        start=date_debut.strftime('%Y-%m-%d'),
        end=date_fin.strftime('%Y-%m-%d'),
        progress=False
    )

    if df_nouveau.empty:
        print(f"  Aucune nouvelle donnée.")
        return df_existant

    if isinstance(df_nouveau.columns, pd.MultiIndex):
        df_nouveau.columns = df_nouveau.columns.get_level_values(0)

    df_nouveau = df_nouveau[['Open', 'High', 'Low', 'Close', 'Volume']]
    df_nouveau.index.name = 'Date'
    df_nouveau.dropna(inplace=True)

    df_final = pd.concat([df_existant, df_nouveau])
    df_final = df_final[~df_final.index.duplicated(keep='last')]
    df_final.sort_index(inplace=True)
    df_final.to_csv(chemin)

    lignes = len(df_final) - len(df_existant)
    print(f"  {lignes} nouvelles lignes ajoutées.")
    print(f"  Total : {len(df_final)} lignes → {chemin}")
    return df_final


def recalculer_features(nom):
    chemin      = f"data/{nom}.csv"
    chemin_feat = f"data/{nom}_features.csv"

    df = pd.read_csv(chemin, index_col=0)
    df.index = pd.to_datetime(df.index, format='mixed')

    df['rendement']   = df['Close'].pct_change()
    df['vol_20j']     = df['rendement'].rolling(20).std()
    df['vol_5j']      = df['rendement'].rolling(5).std()
    df['MA_20']       = df['Close'].rolling(20).mean()
    df['MA_50']       = df['Close'].rolling(50).mean()
    df['MA_ratio']    = df['MA_20'] / df['MA_50']

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    ema12             = df['Close'].ewm(span=12).mean()
    ema26             = df['Close'].ewm(span=26).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    df.dropna(inplace=True)
    df.to_csv(chemin_feat)
    print(f"  Features recalculées → {chemin_feat}")


if __name__ == "__main__":
    print("=" * 50)
    print(f"Mise à jour du {datetime.today().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 50)

    for nom, ticker in actifs.items():
        mettre_a_jour(nom, ticker)
        recalculer_features(nom)

    print("\nMise à jour terminée !")