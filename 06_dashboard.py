import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("Chargement des donnees...")

sp500 = pd.read_csv('data/sp500_features.csv', index_col=0)
sp500.index = pd.to_datetime(sp500.index, format='mixed')

vix = pd.read_csv('data/vix_features.csv', index_col=0)
vix.index = pd.to_datetime(vix.index, format='mixed')

btc = pd.read_csv('data/btc_features.csv', index_col=0)
btc.index = pd.to_datetime(btc.index, format='mixed')

print("Construction du modele...")

df = pd.DataFrame(index=sp500.index)
df['ret_sp500']      = sp500['rendement']
df['vol_sp500']      = sp500['vol_20j']
df['rsi_sp500']      = sp500['RSI']
df['macd_sp500']     = sp500['MACD']
df['ma_ratio_sp500'] = sp500['MA_ratio']

vix_aligned      = vix.reindex(sp500.index, method='ffill')
df['vix_niveau'] = vix_aligned['Close']
df['vix_vol']    = vix_aligned['vol_20j']
df['vix_rsi']    = vix_aligned['RSI']
df.dropna(inplace=True)

def definir_regime(v):
    if v > 30:   return 2
    elif v > 20: return 1
    else:        return 0

df['regime'] = df['vix_niveau'].apply(definir_regime)

features = [
    'ret_sp500', 'vol_sp500', 'rsi_sp500',
    'macd_sp500', 'ma_ratio_sp500',
    'vix_niveau', 'vix_vol', 'vix_rsi'
]

n         = len(df)
train_end = int(n * 0.70)

modele = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
)
modele.fit(
    df[features].iloc[:train_end],
    df['regime'].iloc[:train_end]
)
print("Modele entraine !")

vix_btc      = btc['vol_20j'] * 100 * np.sqrt(252)
seuil_stress = vix_btc.quantile(0.60)
seuil_danger = vix_btc.quantile(0.85)

def regime_btc(v):
    if v > seuil_danger:   return 2
    elif v > seuil_stress: return 1
    else:                  return 0

df_btc = pd.DataFrame(index=btc.index)
df_btc['ret_sp500']      = btc['rendement']
df_btc['vol_sp500']      = btc['vol_20j']
df_btc['rsi_sp500']      = btc['RSI']
df_btc['macd_sp500']     = btc['MACD']
df_btc['ma_ratio_sp500'] = btc['MA_ratio']
df_btc['vix_niveau']     = vix_btc
df_btc['vix_vol']        = vix_btc.rolling(20).std()
df_btc['vix_rsi']        = btc['RSI']
df_btc['Close']          = btc['Close']
df_btc['rendement']      = btc['rendement']
df_btc.dropna(inplace=True)

df_btc['regime']    = df_btc['vix_niveau'].apply(regime_btc)
proba               = modele.predict_proba(df_btc[features])
df_btc['confiance'] = proba.max(axis=1)

derniere       = df_btc.iloc[-1]
regime_auj     = int(derniere['regime'])
confiance_auj  = derniere['confiance'] * 100
prix_actuel    = derniere['Close']
rendement_hier = derniere['rendement'] * 100
vix_actuel     = derniere['vix_niveau']

labels_regime  = {0: 'CALME',  1: 'STRESS',   2: 'DANGER'}
couleurs_regime = {0: '#1D9E75', 1: '#EF9F27', 2: '#E24B4A'}
decisions      = {0: 'TRADE',  1: 'ATTENDRE', 2: 'STOP'}

signal_label = labels_regime[regime_auj]
signal_color = couleurs_regime[regime_auj]
decision     = decisions[regime_auj]

print(f"\nSignal du jour : {signal_label} — {decision}")
print(f"Confiance      : {confiance_auj:.1f}%")
print(f"Prix BTC       : ${prix_actuel:,.0f}")

capital_depart = 10000
capital_ai     = capital_depart
hist_ai        = []

for _, row in df_btc.iterrows():
    if row['regime'] == 0:
        capital_ai = capital_ai * (1 + row['rendement'])
    elif row['regime'] == 1:
        capital_ai = capital_ai * (1 + row['rendement'] * 0.5)
    hist_ai.append(capital_ai)

df_btc['strat_avec_ai'] = hist_ai
df_btc['strat_sans_ai'] = (
    (1 + df_btc['rendement']).cumprod() * capital_depart
)

capital_final_ai   = df_btc['strat_avec_ai'].iloc[-1]
capital_final_sans = df_btc['strat_sans_ai'].iloc[-1]
rendement_ai       = (capital_final_ai / capital_depart - 1) * 100
rendement_sans     = (capital_final_sans / capital_depart - 1) * 100

def sharpe(serie):
    r = serie.pct_change().dropna()
    return (r.mean() / r.std()) * np.sqrt(252)

def max_dd(serie):
    return ((serie - serie.cummax()) / serie.cummax()).min() * 100

sharpe_ai   = sharpe(df_btc['strat_avec_ai'])
sharpe_sans = sharpe(df_btc['strat_sans_ai'])
dd_ai       = max_dd(df_btc['strat_avec_ai'])
dd_sans     = max_dd(df_btc['strat_sans_ai'])

print("\nConstruction du dashboard...")

plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('#0D1117')

gs = gridspec.GridSpec(
    3, 4, figure=fig,
    hspace=0.45, wspace=0.35,
    top=0.92, bottom=0.06,
    left=0.06, right=0.97
)

fig.text(
    0.5, 0.97,
    'Smart Risk Manager — Dashboard',
    ha='center', va='top',
    fontsize=20, fontweight='bold', color='white'
)
fig.text(
    0.5, 0.94,
    f"Mise a jour : {df_btc.index[-1].strftime('%d/%m/%Y')}  |  Bitcoin BTC-USD",
    ha='center', va='top',
    fontsize=11, color='#8B949E'
)

def carte(ax, titre, valeur, sous_titre, couleur_val='white'):
    ax.set_facecolor('#161B22')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.82, titre,
            ha='center', fontsize=10, color='#8B949E')
    ax.text(0.5, 0.52, valeur,
            ha='center', fontsize=20,
            fontweight='bold', color=couleur_val)
    ax.text(0.5, 0.24, sous_titre,
            ha='center', fontsize=10, color='#8B949E')

ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor('#161B22')
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.axis('off')
ax0.add_patch(FancyBboxPatch(
    (0.05, 0.05), 0.9, 0.9,
    boxstyle="round,pad=0.02",
    facecolor=signal_color + '22',
    edgecolor=signal_color,
    linewidth=2
))
ax0.text(0.5, 0.82, 'Signal du jour',
         ha='center', fontsize=10, color='#8B949E')
ax0.text(0.5, 0.58, signal_label,
         ha='center', fontsize=24,
         fontweight='bold', color=signal_color)
ax0.text(0.5, 0.36, decision,
         ha='center', fontsize=15,
         fontweight='bold', color='white')
ax0.text(0.5, 0.16,
         f"Confiance : {confiance_auj:.1f}%",
         ha='center', fontsize=10, color='#8B949E')

signe = '+' if rendement_hier >= 0 else ''
c_rend = '#1D9E75' if rendement_hier >= 0 else '#E24B4A'

carte(fig.add_subplot(gs[0, 1]),
      'Prix Bitcoin',
      f"${prix_actuel:,.0f}",
      f"{signe}{rendement_hier:.2f}% hier",
      couleur_val='white')

carte(fig.add_subplot(gs[0, 2]),
      'Capital final Avec AI',
      f"${capital_final_ai:,.0f}",
      f"Rendement : +{rendement_ai:.1f}%",
      couleur_val='#1D9E75')

carte(fig.add_subplot(gs[0, 3]),
      'Drawdown max',
      f"{dd_ai:.1f}%",
      f"Sans AI : {dd_sans:.1f}%  |  Sharpe : {sharpe_ai:.2f}",
      couleur_val='#1D9E75')

ax_cap = fig.add_subplot(gs[1, :])
ax_cap.set_facecolor('#161B22')
ax_cap.plot(df_btc.index, df_btc['strat_sans_ai'],
            color='#E24B4A', linewidth=1.5,
            label='Sans AI (Buy & Hold)')
ax_cap.plot(df_btc.index, df_btc['strat_avec_ai'],
            color='#1D9E75', linewidth=1.5,
            label='Avec AI (Smart Risk)')
ax_cap.axhline(capital_depart, color='#8B949E',
               linestyle='--', linewidth=0.8, alpha=0.5)
ax_cap.set_title('Evolution du capital — 10 000$ investis',
                 color='white', fontsize=12, pad=8)
ax_cap.tick_params(colors='#8B949E')
ax_cap.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
for spine in ax_cap.spines.values():
    spine.set_edgecolor('#30363D')
ax_cap.legend(
    facecolor='#161B22', edgecolor='#30363D',
    labelcolor='white', fontsize=10
)
ax_cap.grid(True, alpha=0.1, color='#8B949E')

ax_reg = fig.add_subplot(gs[2, :2])
ax_reg.set_facecolor('#161B22')
colors_r = {0: '#1D9E75', 1: '#EF9F27', 2: '#E24B4A'}
for i in range(len(df_btc) - 1):
    r = df_btc['regime'].iloc[i]
    ax_reg.axvspan(
        df_btc.index[i], df_btc.index[i+1],
        alpha=0.25, color=colors_r[r]
    )
ax_reg.plot(df_btc.index, df_btc['Close'],
            color='#58A6FF', linewidth=1)
ax_reg.set_title(
    'Prix BTC par regime   vert=CALME  orange=STRESS  rouge=DANGER',
    color='white', fontsize=10, pad=8
)
ax_reg.tick_params(colors='#8B949E')
ax_reg.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
for spine in ax_reg.spines.values():
    spine.set_edgecolor('#30363D')
ax_reg.grid(True, alpha=0.1, color='#8B949E')

ax_imp = fig.add_subplot(gs[2, 2:])
ax_imp.set_facecolor('#161B22')

importances = pd.Series(
    modele.feature_importances_,
    index=features
).sort_values()

colors_imp = [
    '#1D9E75' if i >= len(importances) - 3 else '#378ADD'
    for i in range(len(importances))
]

bars = ax_imp.barh(
    importances.index,
    importances.values * 100,
    color=colors_imp, alpha=0.85
)
for bar, val in zip(bars, importances.values * 100):
    ax_imp.text(
        val + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.1f}%',
        va='center', color='white', fontsize=9
    )

ax_imp.set_title('Importance des features',
                 color='white', fontsize=11, pad=8)
ax_imp.tick_params(colors='#8B949E')
ax_imp.set_xlabel('Importance (%)', color='#8B949E')
for spine in ax_imp.spines.values():
    spine.set_edgecolor('#30363D')
ax_imp.grid(True, alpha=0.1, axis='x', color='#8B949E')

plt.savefig('data/dashboard.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()
print("\nDashboard sauvegarde → data/dashboard.png")