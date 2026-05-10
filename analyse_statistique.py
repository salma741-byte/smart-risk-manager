# ============================================================
# 3_analyse_statistique.py — VERSION PRO
# Analyse statistique avancée + visualisations professionnelles
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

DB_PATH = "data/market_data.db"

# ============================================================
# STYLE GLOBAL
# ============================================================

plt.style.use('dark_background')

COLORS = {
    'sp500':   '#00FF9F',
    'vix':     '#FF4B4B',
    'bitcoin': '#F7931A',
    'gold':    '#FFD700',
    'dxy':     '#4DA6FF'
}

ACTIFS = ['sp500', 'vix', 'bitcoin', 'gold', 'dxy']

COL_RET   = lambda a: f"{a}_rendement"
COL_CLOSE = lambda a: f"{a}_close"

# ============================================================
# DOSSIERS
# ============================================================

import os
os.makedirs("results/graphs", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        "SELECT * FROM global_dataset",
        conn,
        parse_dates=['date']
    )

    conn.close()

    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    print(f"\nDataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"Période : {df.index[0].date()} → {df.index[-1].date()}")

    return df


# ============================================================
# 1. PERFORMANCE CUMULÉE
# ============================================================

def plot_cumulative_returns(df):

    print("\n[Graphique] Performance cumulée...")

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)

    for actif in ACTIFS:

        col = COL_RET(actif)

        if col not in df.columns:
            continue

        cumulative = (1 + df[col].fillna(0)).cumprod()

        ax.plot(
            cumulative.index,
            cumulative.values,
            label=actif.upper(),
            linewidth=2,
            color=COLORS[actif]
        )

    ax.set_title(
        "Performance Cumulée des Actifs",
        fontsize=20,
        fontweight='bold',
        pad=20
    )

    ax.set_ylabel("Croissance d'un capital de 1$", fontsize=13)

    ax.legend(fontsize=12)

    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/01_cumulative_returns.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# 2. MATRICE DE CORRÉLATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df):

    print("[Graphique] Heatmap corrélations...")

    cols = [COL_RET(a) for a in ACTIFS if COL_RET(a) in df.columns]

    corr = df[cols].corr()

    labels = [c.replace('_rendement', '').upper() for c in cols]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    for i in range(len(labels)):
        for j in range(len(labels)):

            text = ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight='bold'
            )

    ax.set_title(
        "Matrice de Corrélation",
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    fig.colorbar(im)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/02_heatmap_correlations.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# 3. DISTRIBUTION DES RENDEMENTS
# ============================================================

def plot_return_distributions(df):

    print("[Graphique] Distributions statistiques...")

    fig = plt.figure(figsize=(16, 10))

    gs = GridSpec(2, 3, figure=fig)

    positions = [
        (0,0), (0,1), (0,2),
        (1,0), (1,1)
    ]

    for actif, pos in zip(ACTIFS, positions):

        col = COL_RET(actif)

        if col not in df.columns:
            continue

        ax = fig.add_subplot(gs[pos])

        returns = df[col].dropna() * 100

        ax.hist(
            returns,
            bins=60,
            alpha=0.8,
            color=COLORS[actif],
            density=True
        )

        ax.axvline(
            returns.mean(),
            color='white',
            linestyle='--',
            linewidth=2,
            label=f"Moy: {returns.mean():.2f}%"
        )

        ax.set_title(
            actif.upper(),
            fontsize=15,
            fontweight='bold'
        )

        ax.grid(True, alpha=0.2)

        ax.legend()

    plt.suptitle(
        "Distribution des Rendements Journaliers",
        fontsize=20,
        fontweight='bold'
    )

    plt.tight_layout()

    plt.savefig(
        "results/graphs/03_return_distributions.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# 4. VOLATILITÉ ROULANTE
# ============================================================

def plot_rolling_volatility(df):

    print("[Graphique] Volatilité roulante...")

    fig, ax = plt.subplots(figsize=(16, 7))

    for actif in ACTIFS:

        col = COL_RET(actif)

        if col not in df.columns:
            continue

        rolling_vol = (
            df[col]
            .rolling(30)
            .std()
            * np.sqrt(252)
            * 100
        )

        ax.plot(
            rolling_vol.index,
            rolling_vol.values,
            label=actif.upper(),
            linewidth=1.8,
            color=COLORS[actif]
        )

    ax.set_title(
        "Volatilité Annualisée Roulante (30 jours)",
        fontsize=20,
        fontweight='bold'
    )

    ax.set_ylabel("Volatilité %", fontsize=13)

    ax.legend(fontsize=11)

    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/04_rolling_volatility.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# 5. DRAWDOWN S&P500
# ============================================================

def plot_drawdown(df):

    print("[Graphique] Drawdown S&P500...")

    col = COL_RET('sp500')

    returns = df[col].fillna(0)

    cumulative = (1 + returns).cumprod()

    rolling_max = cumulative.cummax()

    drawdown = (cumulative / rolling_max - 1) * 100

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color='red',
        alpha=0.5
    )

    ax.plot(
        drawdown.index,
        drawdown.values,
        color='red',
        linewidth=1.5
    )

    ax.set_title(
        "Drawdown Historique — S&P500",
        fontsize=20,
        fontweight='bold'
    )

    ax.set_ylabel("Drawdown %")

    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/05_drawdown_sp500.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# 6. REGIMES VIX
# ============================================================

def plot_vix_regimes(df):

    print("[Graphique] Régimes VIX...")

    if 'vix_close' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        df.index,
        df['vix_close'],
        color='#FF4B4B',
        linewidth=1.5
    )

    ax.axhline(15, color='green', linestyle='--', alpha=0.8)
    ax.axhline(20, color='orange', linestyle='--', alpha=0.8)
    ax.axhline(30, color='red', linestyle='--', alpha=0.8)

    ax.fill_between(
        df.index,
        df['vix_close'],
        where=df['vix_close'] < 15,
        color='green',
        alpha=0.15,
        label='Très calme'
    )

    ax.fill_between(
        df.index,
        df['vix_close'],
        where=df['vix_close'] > 30,
        color='red',
        alpha=0.20,
        label='Stress extrême'
    )

    ax.set_title(
        "Régimes de Marché basés sur le VIX",
        fontsize=20,
        fontweight='bold'
    )

    ax.set_ylabel("VIX")

    ax.legend()

    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/06_vix_regimes.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# 7. SAISONNALITÉ
# ============================================================

def plot_seasonality(df):

    print("[Graphique] Saisonnalité...")

    col = COL_RET('sp500')

    temp = df.copy()

    temp['month'] = temp.index.month

    monthly = (
        temp.groupby('month')[col]
        .mean()
        * 100
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        monthly.index,
        monthly.values
    )

    ax.set_xticks(range(1, 13))

    ax.set_title(
        "Saisonnalité Mensuelle — S&P500",
        fontsize=18,
        fontweight='bold'
    )

    ax.set_ylabel("Rendement moyen mensuel (%)")

    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/07_seasonality.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# 8. ROLLING CORRELATION SP500 / VIX
# ============================================================

def plot_rolling_correlation(df):

    print("[Graphique] Corrélation glissante SP500/VIX...")

    sp = df[COL_RET('sp500')]
    vx = df[COL_RET('vix')]

    rolling_corr = sp.rolling(60).corr(vx)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        rolling_corr.index,
        rolling_corr.values,
        color='cyan',
        linewidth=2
    )

    ax.axhline(0, color='white', linestyle='--')

    ax.set_title(
        "Corrélation Glissante 60j — S&P500 vs VIX",
        fontsize=18,
        fontweight='bold'
    )

    ax.set_ylabel("Corrélation")

    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/08_rolling_corr_sp_vix.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# 9. DASHBOARD GLOBAL
# ============================================================

def dashboard_global(df):

    print("[Graphique] Dashboard global...")

    fig = plt.figure(figsize=(18, 12))

    gs = GridSpec(3, 2, figure=fig)

    # SP500
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(
        df.index,
        df['sp500_close'],
        color='#00FF9F',
        linewidth=2
    )

    ax1.set_title("S&P500", fontsize=18, fontweight='bold')

    # VIX
    ax2 = fig.add_subplot(gs[1, 0])

    ax2.plot(
        df.index,
        df['vix_close'],
        color='#FF4B4B'
    )

    ax2.set_title("VIX")

    # Bitcoin
    ax3 = fig.add_subplot(gs[1, 1])

    ax3.plot(
        df.index,
        df['bitcoin_close'],
        color='#F7931A'
    )

    ax3.set_title("Bitcoin")

    # Gold
    ax4 = fig.add_subplot(gs[2, 0])

    ax4.plot(
        df.index,
        df['gold_close'],
        color='#FFD700'
    )

    ax4.set_title("Gold")

    # DXY
    ax5 = fig.add_subplot(gs[2, 1])

    ax5.plot(
        df.index,
        df['dxy_close'],
        color='#4DA6FF'
    )

    ax5.set_title("DXY")

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Dashboard Macro-Market",
        fontsize=24,
        fontweight='bold'
    )

    plt.tight_layout()

    plt.savefig(
        "results/graphs/09_dashboard_global.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ANALYSE STATISTIQUE AVANCÉE — VERSION PRO")
    print("=" * 60)

    df = load_dataset()

    # Graphiques
    plot_cumulative_returns(df)
    plot_correlation_heatmap(df)
    plot_return_distributions(df)
    plot_rolling_volatility(df)
    plot_drawdown(df)
    plot_vix_regimes(df)
    plot_seasonality(df)
    plot_rolling_correlation(df)
    dashboard_global(df)

    print("\n" + "=" * 60)
    print("GRAPHIQUES GÉNÉRÉS AVEC SUCCÈS")
    print("=" * 60)

    print("\nDossier : results/graphs/")
    print("""
    01_cumulative_returns.png
    02_heatmap_correlations.png
    03_return_distributions.png
    04_rolling_volatility.png
    05_drawdown_sp500.png
    06_vix_regimes.png
    07_seasonality.png
    08_rolling_corr_sp_vix.png
    09_dashboard_global.png
    """)