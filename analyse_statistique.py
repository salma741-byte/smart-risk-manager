# ============================================================
#  ANALYSE STATISTIQUE COMPLÈTE
#  Comprendre pourquoi le modèle est faible
#  Avant d'améliorer, on doit diagnostiquer
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs("outputs/analyse", exist_ok=True)
DB_PATH = "data/market_data.db"

STYLE = {
    'bg':     '#0d1117', 'bg2': '#161b22', 'border': '#21262d',
    'green':  '#2ecc71', 'red': '#e74c3c', 'blue':   '#3498db',
    'yellow': '#f1c40f', 'text': '#e6edf3', 'muted':  '#7d8590',
    'purple': '#9b59b6', 'cyan': '#1abc9c',
}

def style_dark():
    plt.rcParams.update({
        'figure.facecolor': STYLE['bg'],  'axes.facecolor': STYLE['bg2'],
        'axes.edgecolor':   STYLE['border'], 'axes.labelcolor': STYLE['text'],
        'xtick.color':      STYLE['text'],   'ytick.color':     STYLE['text'],
        'grid.color':       STYLE['border'], 'text.color':      STYLE['text'],
        'legend.facecolor': STYLE['bg'],     'font.family':     'monospace',
    })

def charger_data():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM sp500_ml_features",
                        conn, index_col='date', parse_dates=['date'])
    conn.close()
    df.sort_index(inplace=True)

    # Détecter le régime
    vol_seuil = df['vol_20d'].quantile(0.60)
    cond_vix  = (df['vix_close'] > 20).astype(int)
    cond_vol  = (df['vol_20d'] > vol_seuil).astype(int)
    cond_ma   = (df['close'] < df['ma_200']).astype(int)
    df['regime'] = ((cond_vix + cond_vol + cond_ma) >= 2).astype(int)
    df['regime_label'] = df['regime'].map({0: 'CALME', 1: 'STRESSE'})
    df['ret_demain'] = df['close'].pct_change().shift(-1)
    return df

# ═══════════════════════════════════════════════════════════
# ANALYSE 1 — DISTRIBUTION DES RENDEMENTS PAR RÉGIME
# Question : est-ce que les deux régimes sont vraiment différents ?
# ═══════════════════════════════════════════════════════════
def analyse_distributions(df):
    style_dark()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("ANALYSE 1 — Distribution des rendements par régime",
                 fontsize=14, fontweight='bold')

    calme   = df[df['regime'] == 0]['ret_demain'].dropna() * 100
    stress  = df[df['regime'] == 1]['ret_demain'].dropna() * 100

    # ── Histogrammes superposés ──────────────────────────────────
    ax = axes[0, 0]
    ax.hist(calme,  bins=60, color=STYLE['green'],  alpha=0.6,
            label=f'CALME (n={len(calme)})',   density=True)
    ax.hist(stress, bins=60, color=STYLE['red'],    alpha=0.6,
            label=f'STRESSÉ (n={len(stress)})', density=True)
    ax.axvline(0, color='white', lw=1.5, ls='--')
    ax.axvline(calme.mean(),  color=STYLE['green'], lw=2, ls=':',
               label=f'Moy calme={calme.mean():.3f}%')
    ax.axvline(stress.mean(), color=STYLE['red'],   lw=2, ls=':',
               label=f'Moy stress={stress.mean():.3f}%')
    ax.set_title("Distribution rendements J+1")
    ax.set_xlabel("Rendement (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Test statistique ────────────────────────────────────────
    ax2 = axes[0, 1]
    t_stat, p_val = stats.ttest_ind(calme, stress)
    ks_stat, ks_p = stats.ks_2samp(calme, stress)

    texte = (
        f"TEST T DE STUDENT\n"
        f"─────────────────\n"
        f"t-stat   : {t_stat:.3f}\n"
        f"p-value  : {p_val:.4f}\n"
        f"{'✅ Différence significative (p<0.05)' if p_val < 0.05 else '⚠️ Différence NON significative'}\n\n"
        f"TEST KOLMOGOROV-SMIRNOV\n"
        f"─────────────────────\n"
        f"ks-stat  : {ks_stat:.3f}\n"
        f"p-value  : {ks_p:.4f}\n"
        f"{'✅ Distributions différentes' if ks_p < 0.05 else '⚠️ Distributions similaires'}\n\n"
        f"STATISTIQUES\n"
        f"─────────────────────\n"
        f"Calme   moy : {calme.mean():.4f}%  std : {calme.std():.3f}%\n"
        f"Stressé moy : {stress.mean():.4f}%  std : {stress.std():.3f}%\n"
        f"Ratio vol   : {stress.std()/calme.std():.2f}x plus volatile en stress"
    )
    ax2.text(0.05, 0.95, texte, transform=ax2.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             color=STYLE['text'],
             bbox=dict(boxstyle='round', facecolor=STYLE['bg'], alpha=0.8))
    ax2.axis('off')
    ax2.set_title("Tests statistiques")

    # ── % hausse par mois ────────────────────────────────────────
    ax3 = axes[1, 0]
    monthly = df.groupby(df.index.month)['target'].mean() * 100
    colors  = [STYLE['green'] if v > 50 else STYLE['red'] for v in monthly]
    bars    = ax3.bar(monthly.index, monthly.values, color=colors, alpha=0.85)
    ax3.axhline(50, color='white', lw=1.5, ls='--', label='50% (hasard)')
    mois = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
    ax3.set_xticks(range(1,13)); ax3.set_xticklabels(mois, fontsize=9)
    ax3.set_ylabel("% jours de hausse"); ax3.set_ylim(30, 70)
    ax3.set_title("Saisonnalité — % hausse par mois")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    for bar, val in zip(bars, monthly.values):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.0f}%', ha='center', fontsize=8)

    # ── Autocorrélation des rendements ──────────────────────────
    ax4 = axes[1, 1]
    rets = df['ret_demain'].dropna()
    lags = range(1, 21)
    autocorr = [rets.autocorr(lag=l) for l in lags]
    conf = 1.96 / np.sqrt(len(rets))
    colors_ac = [STYLE['green'] if abs(a) > conf else STYLE['muted'] for a in autocorr]
    ax4.bar(lags, autocorr, color=colors_ac, alpha=0.85)
    ax4.axhline( conf, color='white', lw=1.5, ls='--', label=f'IC 95% (±{conf:.3f})')
    ax4.axhline(-conf, color='white', lw=1.5, ls='--')
    ax4.axhline(0, color='gray', lw=0.8)
    ax4.set_xlabel("Lag (jours)"); ax4.set_ylabel("Autocorrélation")
    ax4.set_title("Autocorrélation des rendements\n(vert = significatif = prédictible)")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/analyse/01_distributions.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("  → outputs/analyse/01_distributions.png")

    # Résumé console
    print(f"\n  RÉSUMÉ DISTRIBUTIONS :")
    print(f"  Différence calme/stress significative : {'OUI ✅' if p_val < 0.05 else 'NON ⚠️'} (p={p_val:.4f})")
    print(f"  Lags significatifs : {[l for l,a in zip(lags,autocorr) if abs(a)>conf]}")


# ═══════════════════════════════════════════════════════════
# ANALYSE 2 — POUVOIR PRÉDICTIF DE CHAQUE FEATURE
# Question : quelles features sont vraiment corrélées avec la cible ?
# ═══════════════════════════════════════════════════════════
def analyse_features(df):
    style_dark()
    EXCLUDE = ['open','high','low','close','volume','inserted_at',
               'target','vix_close','btc_close','regime',
               'regime_label','ret_demain']
    feat_cols = [c for c in df.columns if c not in EXCLUDE
                 and df[c].dtype in [np.float64, np.int64, int, float]]

    # Corrélation point-bisériale avec la cible
    resultats = []
    for col in feat_cols:
        serie = df[col].replace([np.inf,-np.inf], np.nan).dropna()
        idx   = serie.index.intersection(df.index)
        if len(idx) < 100:
            continue
        y = df.loc[idx, 'target']
        try:
            corr, pval = stats.pointbiserialr(serie.loc[idx], y)
            resultats.append({'feature': col, 'corr': corr,
                               'abs_corr': abs(corr), 'pval': pval})
        except:
            pass

    res = pd.DataFrame(resultats).sort_values('abs_corr', ascending=False)
    top20 = res.head(20)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("ANALYSE 2 — Pouvoir prédictif des features",
                 fontsize=14, fontweight='bold')

    # ── Top 20 features par corrélation ─────────────────────────
    ax = axes[0]
    colors = [STYLE['green'] if c > 0 else STYLE['red'] for c in top20['corr']]
    bars   = ax.barh(range(len(top20)), top20['corr'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['feature'], fontsize=9)
    ax.axvline(0, color='white', lw=1)
    conf_line = 1.96 / np.sqrt(len(df))
    ax.axvline( conf_line, color='yellow', lw=1.5, ls='--', alpha=0.7,
                label=f'IC 95% (±{conf_line:.4f})')
    ax.axvline(-conf_line, color='yellow', lw=1.5, ls='--', alpha=0.7)
    ax.set_xlabel("Corrélation avec target (hausse J+1)")
    ax.set_title("Top 20 features — corrélation avec la cible\n(vert=haussier, rouge=baissier)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Scatter corrélation vs p-value ──────────────────────────
    ax2 = axes[1]
    sig  = res[res['pval'] < 0.05]
    nsig = res[res['pval'] >= 0.05]
    ax2.scatter(nsig['abs_corr'], -np.log10(nsig['pval']+1e-10),
                color=STYLE['muted'], alpha=0.6, s=30, label='Non significatif')
    ax2.scatter(sig['abs_corr'],  -np.log10(sig['pval']+1e-10),
                color=STYLE['green'], alpha=0.8, s=60, label='Significatif (p<0.05)')
    ax2.axhline(-np.log10(0.05), color='yellow', lw=1.5, ls='--',
                label='Seuil p=0.05')
    for _, row in sig.head(8).iterrows():
        ax2.annotate(row['feature'],
                     (row['abs_corr'], -np.log10(row['pval']+1e-10)),
                     fontsize=7, color=STYLE['text'],
                     xytext=(5,5), textcoords='offset points')
    ax2.set_xlabel("Corrélation absolue"); ax2.set_ylabel("-log10(p-value)")
    ax2.set_title("Volcano plot — Signal vs Significance\n(haut-droite = features utiles)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/analyse/02_features_predictives.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("  → outputs/analyse/02_features_predictives.png")

    print(f"\n  TOP 10 FEATURES PRÉDICTIVES :")
    print(f"  {'Feature':<25} {'Corrélation':>12} {'p-value':>10} {'Significatif':>14}")
    print(f"  {'─'*65}")
    for _, row in res.head(10).iterrows():
        sig_str = "✅" if row['pval'] < 0.05 else "⚠️"
        print(f"  {row['feature']:<25} {row['corr']:>12.5f} {row['pval']:>10.4f} {sig_str:>14}")

    features_utiles = res[res['pval'] < 0.05]['feature'].tolist()
    print(f"\n  Features significatives (p<0.05) : {len(features_utiles)}/{len(res)}")
    return features_utiles


# ═══════════════════════════════════════════════════════════
# ANALYSE 3 — POURQUOI LE RECALL EST FAIBLE
# Question : le problème vient-il des données ou du modèle ?
# ═══════════════════════════════════════════════════════════
def analyse_recall(df):
    style_dark()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("ANALYSE 3 — Diagnostic Recall & Precision faibles",
                 fontsize=14, fontweight='bold')

    # ── Overlap des classes ──────────────────────────────────────
    # Si les features ont le même aspect pour hausse et baisse → recall faible
    ax = axes[0, 0]
    hausse = df[df['target'] == 1]['ret_1d'].dropna() * 100
    baisse = df[df['target'] == 0]['ret_1d'].dropna() * 100
    ax.hist(hausse, bins=50, color=STYLE['green'], alpha=0.6,
            label=f'Jour suivant = HAUSSE', density=True)
    ax.hist(baisse, bins=50, color=STYLE['red'],   alpha=0.6,
            label=f'Jour suivant = BAISSE', density=True)
    overlap = min(len(hausse), len(baisse)) / max(len(hausse), len(baisse))
    ax.set_title(f"Overlap des classes sur ret_1d\n(overlap élevé = difficile à séparer)")
    ax.set_xlabel("Rendement J (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Séparabilité des classes sur les top features ────────────
    ax2 = axes[0, 1]
    top_feats = ['rsi_14', 'bb_zscore', 'ret_1d', 'vol_20d', 'vix_ret_1d']
    top_feats = [f for f in top_feats if f in df.columns]
    separabilite = {}
    for feat in top_feats:
        h = df[df['target']==1][feat].dropna()
        b = df[df['target']==0][feat].dropna()
        if len(h) > 10 and len(b) > 10:
            t, p = stats.ttest_ind(h, b)
            # Effect size (Cohen's d)
            d = (h.mean()-b.mean()) / np.sqrt((h.std()**2+b.std()**2)/2)
            separabilite[feat] = abs(d)

    sep_s = pd.Series(separabilite).sort_values(ascending=True)
    colors_sep = [STYLE['green'] if v > 0.1 else STYLE['red'] for v in sep_s]
    ax2.barh(sep_s.index, sep_s.values, color=colors_sep, alpha=0.85)
    ax2.axvline(0.2, color='yellow', lw=1.5, ls='--', label="Cohen's d > 0.2 (petit effet)")
    ax2.axvline(0.5, color=STYLE['cyan'], lw=1.5, ls='--', label="Cohen's d > 0.5 (effet moyen)")
    ax2.set_xlabel("Cohen's d (effet de séparation)")
    ax2.set_title("Séparabilité classes Hausse/Baisse\n(Cohen's d — plus grand = mieux)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # ── Bruit dans les données : R² d'une regression simple ──────
    ax3 = axes[1, 0]
    top_feats_all = ['ret_1d','rsi_14','bb_zscore','macd_hist',
                      'vol_20d','vix_ret_1d','btc_ret_1d','prix_vs_ma50']
    top_feats_all = [f for f in top_feats_all if f in df.columns]
    r2_scores = {}
    for feat in top_feats_all:
        sub = df[[feat,'target']].dropna()
        sub = sub.replace([np.inf,-np.inf], np.nan).dropna()
        if len(sub) > 50:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(sub[[feat]])
            lr = LogisticRegression().fit(X, sub['target'])
            r2_scores[feat] = lr.score(X, sub['target']) - 0.5
            # Score - 0.5 car 0.5 = hasard pur

    r2_s = pd.Series(r2_scores).sort_values(ascending=True)
    colors_r2 = [STYLE['green'] if v > 0.02 else STYLE['red'] for v in r2_s]
    ax3.barh(r2_s.index, r2_s.values * 100, color=colors_r2, alpha=0.85)
    ax3.axvline(0, color='white', lw=1, ls='--')
    ax3.set_xlabel("Accuracy - 50% (% au-dessus du hasard)")
    ax3.set_title("Pouvoir prédictif individuel\n(chaque feature seule vs hasard)")
    ax3.grid(True, alpha=0.3)

    # ── Matrice de corrélation entre features ────────────────────
    ax4 = axes[1, 1]
    feats_corr = [f for f in top_feats_all if f in df.columns]
    corr_mat   = df[feats_corr].corr()
    im = ax4.imshow(corr_mat.values, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax4, fraction=0.046)
    ax4.set_xticks(range(len(feats_corr)))
    ax4.set_yticks(range(len(feats_corr)))
    ax4.set_xticklabels(feats_corr, rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels(feats_corr, fontsize=8)
    for i in range(len(feats_corr)):
        for j in range(len(feats_corr)):
            ax4.text(j, i, f"{corr_mat.iloc[i,j]:.1f}",
                     ha='center', va='center', fontsize=7,
                     color='black' if abs(corr_mat.iloc[i,j])>0.5 else 'white')
    ax4.set_title("Corrélations entre features\n(rouge/vert fort = features redondantes)")

    plt.tight_layout()
    plt.savefig("outputs/analyse/03_recall_diagnostic.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("  → outputs/analyse/03_recall_diagnostic.png")


# ═══════════════════════════════════════════════════════════
# ANALYSE 4 — STABILITÉ DU SIGNAL DANS LE TEMPS
# Question : le signal était-il meilleur avant ? A-t-il dégradé ?
# ═══════════════════════════════════════════════════════════
def analyse_stabilite(df):
    style_dark()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("ANALYSE 4 — Stabilité du signal dans le temps",
                 fontsize=14, fontweight='bold')

    # ── Corrélation rolling ret_1d → target ──────────────────────
    ax = axes[0]
    rolling_corr = df['ret_1d'].rolling(60).corr(df['target'].astype(float))
    conf = 1.96 / np.sqrt(60)
    ax.plot(rolling_corr.index, rolling_corr.values,
            color=STYLE['blue'], lw=1.5, label='Corrélation rolling 60j (ret_1d → target)')
    ax.fill_between(rolling_corr.index,
                    rolling_corr.where(rolling_corr > conf),
                    conf, color=STYLE['green'], alpha=0.4, label='Zone haussière significative')
    ax.fill_between(rolling_corr.index,
                    rolling_corr.where(rolling_corr < -conf),
                    -conf, color=STYLE['red'], alpha=0.4, label='Zone baissière significative')
    ax.axhline(0,     color='white', lw=1, ls='--')
    ax.axhline( conf, color='yellow', lw=1, ls=':', alpha=0.7)
    ax.axhline(-conf, color='yellow', lw=1, ls=':', alpha=0.7)
    ax.set_ylabel("Corrélation"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title("Le signal de ret_1d est-il stable dans le temps ?\n(Si le signal disparaît → feature peu fiable)")

    # ── % hausse rolling par année ────────────────────────────────
    ax2 = axes[1]
    annual = df.groupby(df.index.year).agg(
        pct_hausse=('target', 'mean'),
        n_jours=('target', 'count')
    )
    colors_yr = [STYLE['green'] if v > 0.5 else STYLE['red']
                 for v in annual['pct_hausse']]
    bars = ax2.bar(annual.index, annual['pct_hausse']*100,
                   color=colors_yr, alpha=0.85)
    ax2.axhline(50, color='white', lw=1.5, ls='--', label='50% (hasard)')
    for bar, (yr, row) in zip(bars, annual.iterrows()):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.3,
                 f"{row['pct_hausse']*100:.0f}%\n({row['n_jours']}j)",
                 ha='center', fontsize=8)
    ax2.set_ylabel("% jours de hausse"); ax2.set_ylim(35, 70)
    ax2.set_title("% jours de hausse par année\n(Varie selon les conditions macro — explique la difficulté)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/analyse/04_stabilite_signal.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("  → outputs/analyse/04_stabilite_signal.png")


# ═══════════════════════════════════════════════════════════
# RAPPORT CONSOLE FINAL
# ═══════════════════════════════════════════════════════════
def rapport_final(df, features_utiles):
    print("\n" + "═"*60)
    print("  DIAGNOSTIC FINAL — POURQUOI LE MODÈLE EST LIMITÉ")
    print("═"*60)

    # Test si la cible est prédictible
    rets = df['ret_demain'].dropna()
    _, p_norm = stats.normaltest(rets)
    acf1 = rets.autocorr(lag=1)

    print(f"""
  1. NATURE DES DONNÉES
  ─────────────────────
  • Distribution normale : {'NON (leptokurtique)' if p_norm < 0.05 else 'OUI'}
  • Autocorrélation J+1  : {acf1:.4f}
    → {'Signal faible mais réel' if abs(acf1) > 0.02 else 'Quasiment aucune mémoire'}

  2. FEATURES SIGNIFICATIVES
  ──────────────────────────
  • {len(features_utiles)} features sur {df.shape[1]} ont p<0.05
  • Les plus utiles : {', '.join(features_utiles[:5])}

  3. POURQUOI LE RECALL BAISSE EST FAIBLE
  ────────────────────────────────────────
  • Le S&P monte {df['target'].mean()*100:.1f}% du temps → biais haussier naturel
  • Le modèle apprend à prédire "hausse" par défaut
  • Solution : class_weight + seuil optimisé ✅ (déjà fait)

  4. POURQUOI L'ACCURACY PLAFONNE À ~55-63%
  ──────────────────────────────────────────
  • Le marché est quasi-efficient sur J+1
  • Les indicateurs techniques sont publics → déjà intégrés dans les prix
  • La volatilité domine le signal en période calme
  • Maximum théorique avec données prix seules : ~58-62%

  5. RECOMMANDATIONS PRIORITAIRES
  ────────────────────────────────
  ✅ Déjà fait    : Régime switching, class_weight, seuil optimal
  🔧 À faire      : Fear & Greed Index (forward-looking)
  🔧 À faire      : Allonger l'horizon de prédiction (J+3 ou J+5)
  🔧 À faire      : Purged cross-validation (éviter la fuite temporelle)
    """)


# ─────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ANALYSE STATISTIQUE COMPLÈTE")
    print("=" * 55)

    df = charger_data()
    print(f"Dataset : {len(df)} jours, {df.shape[1]} colonnes\n")

    print("Analyse 1 — Distributions...")
    analyse_distributions(df)

    print("\nAnalyse 2 — Pouvoir prédictif des features...")
    features_utiles = analyse_features(df)

    print("\nAnalyse 3 — Diagnostic recall/precision...")
    analyse_recall(df)

    print("\nAnalyse 4 — Stabilité du signal...")
    analyse_stabilite(df)

    rapport_final(df, features_utiles)

    print("\n  Graphiques sauvegardés dans outputs/analyse/")
    print("  Lance : explorer outputs\\analyse")