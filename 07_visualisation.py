# ============================================================
#  MODULE 4 — ANALYSE STATISTIQUE & VISUALISATIONS
#  Graphiques de performance, courbes de backtest,
#  analyse des signaux, heatmaps de features
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs("outputs", exist_ok=True)
DB_PATH = "data/market_data.db"

COULEURS = {
    'ACHAT':  '#2ecc71',
    'VENTE':  '#e74c3c',
    'NEUTRE': '#f39c12',
    'bh':     '#3498db',
    'strat':  '#9b59b6',
    'fond':   '#0d1117',
    'grille': '#21262d',
    'texte':  '#c9d1d9',
}

def style_dark():
    plt.rcParams.update({
        'figure.facecolor':  COULEURS['fond'],
        'axes.facecolor':    COULEURS['fond'],
        'axes.edgecolor':    COULEURS['grille'],
        'axes.labelcolor':   COULEURS['texte'],
        'xtick.color':       COULEURS['texte'],
        'ytick.color':       COULEURS['texte'],
        'grid.color':        COULEURS['grille'],
        'text.color':        COULEURS['texte'],
        'legend.facecolor':  '#161b22',
        'legend.edgecolor':  COULEURS['grille'],
        'font.family':       'monospace',
    })

# ─────────────────────────────────────────
# 1. COURBES DE PERFORMANCE BACKTEST
# ─────────────────────────────────────────
def plot_performance_backtest():
    bt = pd.read_csv("data/signaux_historiques.csv",
                     index_col='date', parse_dates=['date'])

    bt['ret_bh']    = bt['prix'].pct_change()
    bt['ret_strat'] = bt['position'] * bt['ret_bh']
    bt.dropna(inplace=True)
    cum_bh    = (1 + bt['ret_bh']).cumprod()
    cum_strat = (1 + bt['ret_strat']).cumprod()

    style_dark()
    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                              gridspec_kw={'height_ratios': [3,1.5,1]})
    fig.suptitle("S&P 500 — Backtest Stratégie Ensemble ML",
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Courbes cumulées ─────────────────────────────────────
    ax = axes[0]
    ax.plot(cum_bh.index,    cum_bh.values,    color=COULEURS['bh'],
            lw=1.8, label='Buy & Hold', alpha=0.9)
    ax.plot(cum_strat.index, cum_strat.values, color=COULEURS['strat'],
            lw=2.2, label='Stratégie ML', alpha=0.9)

    # Zones de signaux
    for i in range(len(bt)):
        s = bt['signal'].iloc[i]
        if s != 'NEUTRE':
            ax.axvspan(bt.index[i],
                       bt.index[i+1] if i+1 < len(bt) else bt.index[i],
                       color=COULEURS[s], alpha=0.07, lw=0)

    ax.set_ylabel("Performance cumulée (base 1)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.set_title("Performance cumulée", fontsize=12, pad=8)

    # ── Drawdown ─────────────────────────────────────────────
    ax2 = axes[1]
    dd_bh    = (cum_bh    / cum_bh.cummax()    - 1) * 100
    dd_strat = (cum_strat / cum_strat.cummax() - 1) * 100
    ax2.fill_between(dd_bh.index,    dd_bh,    0,
                     color=COULEURS['bh'],   alpha=0.45,
                     label='Drawdown BH')
    ax2.fill_between(dd_strat.index, dd_strat, 0,
                     color=COULEURS['strat'], alpha=0.55,
                     label='Drawdown Strat')
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4)
    ax2.set_title("Drawdown", fontsize=12, pad=8)

    # ── Probabilité journalière ───────────────────────────────
    ax3 = axes[2]
    colors_bar = [COULEURS[s] for s in bt['signal']]
    ax3.bar(bt.index, bt['proba'], color=colors_bar,
            width=1, alpha=0.75)
    ax3.axhline(0.60, color=COULEURS['ACHAT'], lw=1.2,
                ls='--', label='Seuil achat (60%)')
    ax3.axhline(0.40, color=COULEURS['VENTE'], lw=1.2,
                ls='--', label='Seuil vente (40%)')
    ax3.set_ylabel("P(hausse)", fontsize=11)
    ax3.legend(fontsize=9, ncol=2)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.4)
    ax3.set_title("Probabilité de hausse journalière", fontsize=12, pad=8)

    legend_patches = [
        Patch(color=COULEURS['ACHAT'],  label='ACHAT'),
        Patch(color=COULEURS['VENTE'],  label='VENTE'),
        Patch(color=COULEURS['NEUTRE'], label='NEUTRE'),
    ]
    fig.legend(handles=legend_patches, loc='lower right',
               fontsize=10, ncol=3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig("outputs/backtest_performance.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → outputs/backtest_performance.png")

# ─────────────────────────────────────────
# 2. ANALYSE STATISTIQUE DES SIGNAUX
# ─────────────────────────────────────────
def plot_analyse_signaux():
    bt = pd.read_csv("data/signaux_historiques.csv",
                     index_col='date', parse_dates=['date'])
    bt['ret_bh']  = bt['prix'].pct_change()
    bt['ret_next'] = bt['ret_bh'].shift(-1)
    bt.dropna(inplace=True)

    style_dark()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Analyse Statistique des Signaux ML", fontsize=15,
                 fontweight='bold')

    # ── Distribution des rendements par signal ────────────────
    ax = axes[0, 0]
    for sig, col in [('ACHAT', COULEURS['ACHAT']),
                     ('VENTE', COULEURS['VENTE']),
                     ('NEUTRE', COULEURS['NEUTRE'])]:
        data = bt[bt['signal'] == sig]['ret_next'] * 100
        if len(data) > 0:
            ax.hist(data, bins=40, color=col, alpha=0.6,
                    label=f'{sig} (n={len(data)})', density=True)
    ax.axvline(0, color='white', lw=1, ls='--')
    ax.set_xlabel("Rendement J+1 (%)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des rendements par signal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Win rate mensuel ──────────────────────────────────────
    ax2 = axes[0, 1]
    bt_achat = bt[bt['signal'] == 'ACHAT'].copy()
    bt_achat['year_month'] = bt_achat.index.to_period('M')
    monthly  = bt_achat.groupby('year_month').apply(
        lambda g: (g['ret_next'] > 0).mean()
    )
    colors_m = [COULEURS['ACHAT'] if v >= 0.5 else COULEURS['VENTE']
                for v in monthly]
    ax2.bar(range(len(monthly)), monthly.values, color=colors_m, alpha=0.8)
    ax2.axhline(0.5, color='white', lw=1.2, ls='--', label='50%')
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("Win rate")
    ax2.set_title("Win rate mensuel (signaux ACHAT)")
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    ax2.set_xticklabels(
        [str(monthly.index[i]) for i in range(0, len(monthly),
                                               max(1, len(monthly)//12))],
        rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Rendement moyen par signal ────────────────────────────
    ax3 = axes[1, 0]
    stats = bt.groupby('signal')['ret_next'].agg(['mean','std','count'])
    stats['mean_pct'] = stats['mean'] * 100
    stats['std_pct']  = stats['std']  * 100
    cols_bar = [COULEURS.get(s, 'gray') for s in stats.index]
    bars = ax3.bar(stats.index, stats['mean_pct'], color=cols_bar, alpha=0.85)
    ax3.errorbar(stats.index, stats['mean_pct'], yerr=stats['std_pct'],
                 fmt='none', color='white', capsize=5, lw=1.5)
    ax3.axhline(0, color='white', lw=1, ls='--')
    ax3.set_ylabel("Rendement moyen J+1 (%)")
    ax3.set_title("Rendement moyen par type de signal")
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax3.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 0.01,
                 f"n={int(row['count'])}", ha='center',
                 va='bottom', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Probabilité vs rendement réel ─────────────────────────
    ax4 = axes[1, 1]
    bt['proba_bin'] = pd.cut(bt['proba'], bins=10)
    cal = bt.groupby('proba_bin').apply(
        lambda g: pd.Series({
            'mean_proba':  g['proba'].mean(),
            'mean_ret_pos': (g['ret_next'] > 0).mean()
        })
    )
    ax4.scatter(cal['mean_proba'], cal['mean_ret_pos'],
                color=COULEURS['strat'], s=80, zorder=5)
    ax4.plot([0, 1], [0, 1], 'w--', lw=1.2, label='Calibration parfaite')
    ax4.set_xlabel("Probabilité prédite")
    ax4.set_ylabel("Fréquence réelle de hausse")
    ax4.set_title("Courbe de calibration du modèle")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("outputs/analyse_signaux.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → outputs/analyse_signaux.png")

# ─────────────────────────────────────────
# 3. HEATMAP CORRÉLATION FEATURES
# ─────────────────────────────────────────
def plot_correlation_features():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sp500_ml_features",
                     conn, index_col='date', parse_dates=['date'])
    conn.close()

    EXCLUDE = ['open','high','low','volume','inserted_at',
               'vix_close','btc_close']
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in EXCLUDE][:20]  # top 20

    corr = df[num_cols].corr()

    style_dark()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(num_cols, fontsize=8)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.1f}",
                    ha='center', va='center', fontsize=6.5,
                    color='black' if abs(corr.iloc[i,j]) > 0.5 else COULEURS['texte'])
    ax.set_title("Matrice de corrélation des features",
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("outputs/correlation_features.png",
                dpi=130, bbox_inches='tight')
    plt.close()
    print("  → outputs/correlation_features.png")

# ─────────────────────────────────────────
# 4. DASHBOARD RÉSUMÉ DERNIER SIGNAL
# ─────────────────────────────────────────
def plot_dashboard_signal():
    bt = pd.read_csv("data/signaux_historiques.csv",
                     index_col='date', parse_dates=['date'])
    bt['ret_bh'] = bt['prix'].pct_change()

    last        = bt.iloc[-1]
    signal_last = last['signal']
    proba_last  = last['proba']
    date_last   = bt.index[-1].strftime('%d/%m/%Y')

    style_dark()
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle("DASHBOARD — S&P 500 Predictor",
                 fontsize=17, fontweight='bold')

    # ── Signal central ───────────────────────────────────────
    ax_sig = fig.add_subplot(gs[:, 0])
    ax_sig.set_xlim(0, 1); ax_sig.set_ylim(0, 1)
    ax_sig.axis('off')
    col_sig = COULEURS.get(signal_last, 'white')
    ax_sig.add_patch(plt.Circle((0.5, 0.55), 0.35,
                                 color=col_sig, alpha=0.2))
    ax_sig.text(0.5, 0.55, signal_last, ha='center', va='center',
                fontsize=28, fontweight='bold', color=col_sig)
    ax_sig.text(0.5, 0.18,
                f"P(hausse) = {proba_last*100:.1f}%",
                ha='center', fontsize=14, color=COULEURS['texte'])
    ax_sig.text(0.5, 0.08,
                f"Données au {date_last}",
                ha='center', fontsize=10, color='gray')
    ax_sig.set_title("Signal pour demain", fontsize=12, pad=10)

    # ── Prix récent S&P 500 ──────────────────────────────────
    ax_prix = fig.add_subplot(gs[0, 1:])
    recent  = bt['prix'].iloc[-60:]
    ax_prix.plot(recent.index, recent.values,
                 color=COULEURS['bh'], lw=2)
    # Colorier les 30 derniers jours selon signal
    for i in range(max(0, len(bt)-30), len(bt)-1):
        s   = bt['signal'].iloc[i]
        ax_prix.axvspan(bt.index[i], bt.index[i+1],
                        color=COULEURS[s], alpha=0.15)
    ax_prix.set_title("S&P 500 — 60 derniers jours (zones = signaux)",
                      fontsize=11)
    ax_prix.set_ylabel("Prix ($)")
    ax_prix.grid(True, alpha=0.35)

    # ── Probabilités 30 derniers jours ─────────────────────
    ax_prob = fig.add_subplot(gs[1, 1:])
    recent_30 = bt.iloc[-30:]
    cols_bar  = [COULEURS[s] for s in recent_30['signal']]
    ax_prob.bar(recent_30.index, recent_30['proba'],
                color=cols_bar, width=1, alpha=0.8)
    ax_prob.axhline(0.60, color=COULEURS['ACHAT'], lw=1.2,
                    ls='--', alpha=0.8)
    ax_prob.axhline(0.40, color=COULEURS['VENTE'], lw=1.2,
                    ls='--', alpha=0.8)
    ax_prob.axhline(0.50, color='white',           lw=0.8,
                    ls=':', alpha=0.5)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("P(hausse)")
    ax_prob.set_title("Probabilités — 30 derniers jours", fontsize=11)
    ax_prob.grid(True, alpha=0.35)

    plt.savefig("outputs/dashboard_signal.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → outputs/dashboard_signal.png")

# ─────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MODULE 4 — ANALYSE STATISTIQUE & VISUALISATIONS")
    print("=" * 55)

    print("Génération des graphiques...")
    plot_performance_backtest()
    plot_analyse_signaux()
    plot_correlation_features()
    plot_dashboard_signal()

    print("\nTous les graphiques sauvegardés dans /outputs/")
    print("Module 4 terminé ✓")