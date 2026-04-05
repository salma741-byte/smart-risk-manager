import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("BACKTESTING — SANS AI vs AVEC AI")
print("=" * 50)

# ─────────────────────────────────────────
# 1. CHARGER LES DONNEES
# ─────────────────────────────────────────
sp500 = pd.read_csv('data/sp500_features.csv', index_col=0)
sp500.index = pd.to_datetime(sp500.index, format='mixed')

vix = pd.read_csv('data/vix_features.csv', index_col=0)
vix.index = pd.to_datetime(vix.index, format='mixed')

btc = pd.read_csv('data/btc_features.csv', index_col=0)
btc.index = pd.to_datetime(btc.index, format='mixed')

# ─────────────────────────────────────────
# 2. RECREER LE MODELE
# ─────────────────────────────────────────
df = pd.DataFrame(index=sp500.index)
df['ret_sp500']      = sp500['rendement']
df['vol_sp500']      = sp500['vol_20j']
df['rsi_sp500']      = sp500['RSI']
df['macd_sp500']     = sp500['MACD']
df['ma_ratio_sp500'] = sp500['MA_ratio']

vix_aligned          = vix.reindex(sp500.index, method='ffill')
df['vix_niveau']     = vix_aligned['Close']
df['vix_vol']        = vix_aligned['vol_20j']
df['vix_rsi']        = vix_aligned['RSI']
df.dropna(inplace=True)

def definir_regime(v):
    if v > 30:   return 2
    elif v > 20: return 1
    else:        return 0

df['regime'] = df['vix_niveau'].apply(definir_regime)

features = ['ret_sp500', 'vol_sp500', 'rsi_sp500',
            'macd_sp500', 'ma_ratio_sp500',
            'vix_niveau', 'vix_vol', 'vix_rsi']

n         = len(df)
train_end = int(n * 0.70)

modele = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    random_state=42, class_weight='balanced'
)
modele.fit(df[features].iloc[:train_end],
           df['regime'].iloc[:train_end])

print("Modele entraine sur S&P500 + VIX")

# ─────────────────────────────────────────
# 3. PREPARER LES DONNEES BITCOIN
# ─────────────────────────────────────────
df_btc = pd.DataFrame(index=btc.index)
df_btc['ret_sp500']      = btc['rendement']
df_btc['vol_sp500']      = btc['vol_20j']
df_btc['rsi_sp500']      = btc['RSI']
df_btc['macd_sp500']     = btc['MACD']
df_btc['ma_ratio_sp500'] = btc['MA_ratio']

# VIX synthetique adapte a Bitcoin
vix_btc = btc['vol_20j'] * 100 * np.sqrt(252)

# Seuils relatifs bases sur la distribution reelle de BTC
seuil_stress = vix_btc.quantile(0.60)
seuil_danger = vix_btc.quantile(0.85)

print(f"\nSeuils BTC adaptes :")
print(f"  CALME  si VIX < {seuil_stress:.1f}")
print(f"  STRESS si VIX entre {seuil_stress:.1f} et {seuil_danger:.1f}")
print(f"  DANGER si VIX > {seuil_danger:.1f}")

df_btc['vix_niveau'] = vix_btc
df_btc['vix_vol']    = df_btc['vix_niveau'].rolling(20).std()
df_btc['vix_rsi']    = btc['RSI']
df_btc['Close']      = btc['Close']
df_btc['rendement']  = btc['rendement']
df_btc.dropna(inplace=True)

# Regime adapte a Bitcoin
def regime_btc(v):
    if v > seuil_danger:   return 2
    elif v > seuil_stress: return 1
    else:                  return 0

df_btc['regime']    = df_btc['vix_niveau'].apply(regime_btc)
df_btc['confiance'] = modele.predict_proba(
    df_btc[features]
).max(axis=1)

print(f"\nDistribution regimes BTC :")
print(f"  CALME  : {(df_btc['regime']==0).sum()} jours ({(df_btc['regime']==0).mean()*100:.1f}%)")
print(f"  STRESS : {(df_btc['regime']==1).sum()} jours ({(df_btc['regime']==1).mean()*100:.1f}%)")
print(f"  DANGER : {(df_btc['regime']==2).sum()} jours ({(df_btc['regime']==2).mean()*100:.1f}%)")

# ─────────────────────────────────────────
# 4. STRATEGIE SANS AI — Buy and Hold
#    On reste investi tout le temps
# ─────────────────────────────────────────
capital_depart = 10000

df_btc['strat_sans_ai'] = (
    (1 + df_btc['rendement']).cumprod() * capital_depart
)

# ─────────────────────────────────────────
# 5. STRATEGIE AVEC AI
#    CALME  → investi 100%
#    STRESS → investi 50%
#    DANGER → cash (0%)
# ─────────────────────────────────────────
capital    = capital_depart
historique = []

for _, row in df_btc.iterrows():
    if row['regime'] == 0:
        # CALME → pleinement investi
        capital = capital * (1 + row['rendement'])
    elif row['regime'] == 1:
        # STRESS → moitie investi
        capital = capital * (1 + row['rendement'] * 0.5)
    else:
        # DANGER → cash, on ne perd rien
        capital = capital
    historique.append(capital)

df_btc['strat_avec_ai'] = historique

# ─────────────────────────────────────────
# 6. METRIQUES DE PERFORMANCE
# ─────────────────────────────────────────
def calculer_metriques(serie, capital_depart):
    rendements      = serie.pct_change().dropna()
    rendement_total = (serie.iloc[-1] / capital_depart - 1) * 100
    peak            = serie.cummax()
    drawdown        = (serie - peak) / peak
    max_dd          = drawdown.min() * 100
    sharpe          = (rendements.mean() / rendements.std()) * np.sqrt(252)
    vol             = rendements.std() * np.sqrt(252) * 100
    return {
        'Rendement total'    : f"{rendement_total:.1f}%",
        'Capital final'      : f"${serie.iloc[-1]:,.0f}",
        'Drawdown max'       : f"{max_dd:.1f}%",
        'Sharpe ratio'       : f"{sharpe:.2f}",
        'Volatilite annuelle': f"{vol:.1f}%"
    }

m_sans = calculer_metriques(df_btc['strat_sans_ai'], capital_depart)
m_avec = calculer_metriques(df_btc['strat_avec_ai'], capital_depart)

print(f"\n{'Metrique':<22} {'SANS AI':>12} {'AVEC AI':>12}")
print("-" * 48)
for k in m_sans:
    print(f"{k:<22} {m_sans[k]:>12} {m_avec[k]:>12}")

# ─────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Smart Risk Manager — Backtesting Bitcoin',
             fontsize=16, fontweight='bold')

# Graphique 1 : Evolution du capital
ax1 = axes[0]
ax1.plot(df_btc.index, df_btc['strat_sans_ai'],
         color='#E24B4A', linewidth=1.5,
         label='Sans AI (Buy & Hold)')
ax1.plot(df_btc.index, df_btc['strat_avec_ai'],
         color='#1D9E75', linewidth=1.5,
         label='Avec AI (Smart Risk)')
ax1.axhline(capital_depart, color='gray', linestyle='--',
            linewidth=0.8, alpha=0.6, label='Capital depart')
ax1.set_title('Evolution du capital — 10 000$ investis')
ax1.set_ylabel('Capital ($)')
ax1.legend()
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.grid(True, alpha=0.2)

# Graphique 2 : Prix BTC + regimes
ax2 = axes[1]
colors_regime = {0: '#1D9E75', 1: '#EF9F27', 2: '#E24B4A'}
for i in range(len(df_btc) - 1):
    r = df_btc['regime'].iloc[i]
    ax2.axvspan(df_btc.index[i], df_btc.index[i+1],
                alpha=0.25, color=colors_regime[r])
ax2.plot(df_btc.index, df_btc['Close'],
         color='#185FA5', linewidth=1, label='Prix BTC')
ax2.set_title('Prix Bitcoin — vert=CALME  orange=STRESS  rouge=DANGER')
ax2.set_ylabel('Prix ($)')
ax2.legend()
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax2.grid(True, alpha=0.2)

# Graphique 3 : Drawdown
ax3 = axes[2]
for col, color, label in [
    ('strat_sans_ai', '#E24B4A', 'Sans AI'),
    ('strat_avec_ai', '#1D9E75', 'Avec AI')
]:
    peak = df_btc[col].cummax()
    dd   = (df_btc[col] - peak) / peak * 100
    ax3.fill_between(df_btc.index, dd, 0,
                     alpha=0.4, color=color, label=label)
    ax3.plot(df_btc.index, dd, color=color, linewidth=0.8)

ax3.set_title('Drawdown — Perte maximale depuis le sommet')
ax3.set_ylabel('Drawdown (%)')
ax3.legend()
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('data/backtest_result.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGraphique sauvegarde → data/backtest_result.png")
print("Backtesting termine !")