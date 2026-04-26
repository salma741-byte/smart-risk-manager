# ============================================================
#  7_backtest.py — BACKTEST COMPLET
#  Teste la stratégie sur données historiques
#  Compare vs Buy & Hold
#  Génère un rapport HTML interactif
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs", exist_ok=True)
DB_PATH = "data/market_data.db"

# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────
def charger_donnees():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM sp500_ml_features",
                        conn, index_col='date', parse_dates=['date'])
    conn.close()
    df.sort_index(inplace=True)
    return df

def charger_modeles():
    try:
        params    = joblib.load("models/regime_params.pkl")
        modeles   = {}
        for nom in ["stresse", "calme_tendance", "calme_lateral"]:
            try:
                modeles[nom] = {
                    'ensemble': joblib.load(f"models/ensemble_{nom}.pkl"),
                    'scaler':   joblib.load(f"models/scaler_{nom}.pkl"),
                    'feats':    joblib.load(f"models/feats_{nom}.pkl"),
                    'seuil':    joblib.load(f"models/seuil_{nom}.pkl"),
                }
            except:
                pass
        return modeles, params
    except Exception as e:
        print(f"  ⚠️  Modèles non trouvés : {e}")
        return {}, {}

# ─────────────────────────────────────────
# 2. DÉTECTION RÉGIME
# ─────────────────────────────────────────
def detecter_regime_ligne(row, vol_seuil, vix_seuil=20.0):
    cond_vix = int(row.get('vix_close', 15) > vix_seuil)
    cond_vol = int(row.get('vol_20d',  0.01) > vol_seuil)
    cond_ma  = int(row.get('close', 1) < row.get('ma_200', 1))
    score    = cond_vix + cond_vol + cond_ma
    if score >= 2:
        return "stresse"
    elif row.get('adx_14', 0) > 20:
        return "calme_tendance"
    else:
        return "calme_lateral"

# ─────────────────────────────────────────
# 3. PRÉDICTION SUR TOUT L'HISTORIQUE
# ─────────────────────────────────────────
def predire_historique(df, modeles, params):
    vol_seuil  = params.get('vol_seuil', df['vol_20d'].quantile(0.60))
    vix_seuil  = params.get('vix_seuil', 20.0)
    confiance  = params.get('confiance', {
        "STRESSE":        {'achat': 0.58, 'vente': 0.42},
        "CALME_TENDANCE": {'achat': 0.60, 'vente': 0.40},
        "CALME_LATERAL":  {'achat': 0.62, 'vente': 0.38},
    })

    probas  = []
    signaux = []
    regimes = []

    print(f"  Calcul des prédictions sur {len(df)} jours...")
    for i, (date, row) in enumerate(df.iterrows()):
        regime_key = detecter_regime_ligne(row, vol_seuil, vix_seuil)
        regimes.append(regime_key.upper())

        if regime_key not in modeles:
            # Fallback
            for fallback in ["calme_tendance", "stresse"]:
                if fallback in modeles:
                    regime_key = fallback
                    break
            else:
                probas.append(0.5)
                signaux.append("NEUTRE")
                continue

        m     = modeles[regime_key]
        feats = [f for f in m['feats'] if f in df.columns]
        X     = np.array([[row.get(f, 0) for f in feats]])
        X     = np.nan_to_num(X)

        try:
            X_s   = np.nan_to_num(m['scaler'].transform(X))
            p_xgb = m['ensemble']['xgb'].predict_proba(X_s)[0, 1]
            p_rf  = m['ensemble']['rf'].predict_proba(X_s)[0, 1]
            proba = m['ensemble']['w_xgb'] * p_xgb + m['ensemble']['w_rf'] * p_rf
        except:
            proba = 0.5

        probas.append(proba)

        regime_upper = regime_key.upper().replace('_TENDANCE','_TENDANCE').replace('_LATERAL','_LATERAL')
        conf = confiance.get(regime_key.upper().replace('stresse','STRESSE')
                              .replace('calme_tendance','CALME_TENDANCE')
                              .replace('calme_lateral','CALME_LATERAL'),
                              {'achat': 0.60, 'vente': 0.40})

        if proba >= conf['achat']:
            signaux.append("ACHAT")
        elif proba <= conf['vente']:
            signaux.append("VENTE")
        else:
            signaux.append("NEUTRE")

    df['proba']   = probas
    df['signal']  = signaux
    df['regime']  = regimes
    return df

# ─────────────────────────────────────────
# 4. SIMULATION BACKTEST
# ─────────────────────────────────────────
def simuler_backtest(df, frais=0.001):
    """
    Stratégie long-only :
    - ACHAT  → position 1 (investi)
    - VENTE  → position -1 (short léger, 50%)
    - NEUTRE → position 0 (cash)
    Frais : 0.1% par transaction
    """
    bt = df.copy()
    bt['ret_daily'] = bt['close'].pct_change()

    # Position décalée de 1 jour (on agit le lendemain du signal)
    bt['position'] = bt['signal'].shift(1).map(
        {'ACHAT': 1.0, 'NEUTRE': 0.0, 'VENTE': -0.5}
    ).fillna(0)

    # Frais sur changement de position
    bt['pos_change'] = bt['position'].diff().abs()
    bt['frais_app']  = bt['pos_change'] * frais

    # Rendement stratégie
    bt['ret_strat'] = bt['position'] * bt['ret_daily'] - bt['frais_app']

    # Cumul
    bt['cum_bh']    = (1 + bt['ret_daily']).cumprod()
    bt['cum_strat'] = (1 + bt['ret_strat']).cumprod()

    return bt.dropna()

# ─────────────────────────────────────────
# 5. MÉTRIQUES FINANCIÈRES
# ─────────────────────────────────────────
def calculer_metriques(bt):
    rf = 0.02 / 252  # taux sans risque journalier

    def sharpe(rets):
        excess = rets - rf
        return (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0

    def max_drawdown(cum):
        dd = (cum / cum.cummax() - 1)
        return dd.min()

    def calmar(rets, cum):
        ann_ret = (cum.iloc[-1] ** (252/len(cum))) - 1
        mdd     = abs(max_drawdown(cum))
        return ann_ret / mdd if mdd > 0 else 0

    def sortino(rets):
        excess   = rets - rf
        neg_rets = excess[excess < 0]
        downside = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else 1
        return excess.mean() * 252 / downside

    n_jours     = len(bt)
    ann_factor  = 252 / n_jours

    ret_bh      = float(bt['cum_bh'].iloc[-1] - 1)
    ret_strat   = float(bt['cum_strat'].iloc[-1] - 1)
    ann_bh      = float((bt['cum_bh'].iloc[-1] ** ann_factor) - 1)
    ann_strat   = float((bt['cum_strat'].iloc[-1] ** ann_factor) - 1)

    metriques = {
        # Returns
        'ret_total_bh':    round(ret_bh * 100, 2),
        'ret_total_strat': round(ret_strat * 100, 2),
        'ret_ann_bh':      round(ann_bh * 100, 2),
        'ret_ann_strat':   round(ann_strat * 100, 2),
        # Risque
        'sharpe_bh':       round(sharpe(bt['ret_daily']), 3),
        'sharpe_strat':    round(sharpe(bt['ret_strat']), 3),
        'sortino_bh':      round(sortino(bt['ret_daily']), 3),
        'sortino_strat':   round(sortino(bt['ret_strat']), 3),
        'mdd_bh':          round(max_drawdown(bt['cum_bh']) * 100, 2),
        'mdd_strat':       round(max_drawdown(bt['cum_strat']) * 100, 2),
        'calmar_bh':       round(calmar(bt['ret_daily'], bt['cum_bh']), 3),
        'calmar_strat':    round(calmar(bt['ret_strat'], bt['cum_strat']), 3),
        # Signaux
        'n_achat':         int((bt['signal'] == 'ACHAT').sum()),
        'n_vente':         int((bt['signal'] == 'VENTE').sum()),
        'n_neutre':        int((bt['signal'] == 'NEUTRE').sum()),
        'win_rate':        round(float((bt['ret_strat'] > 0).mean()) * 100, 2),
        'jours_investis':  round(float((bt['position'] != 0).mean()) * 100, 2),
        'n_trades':        int(bt['pos_change'].sum()),
        # Périodes
        'date_debut':      bt.index[0].strftime('%Y-%m-%d'),
        'date_fin':        bt.index[-1].strftime('%Y-%m-%d'),
        'n_jours':         n_jours,
    }
    return metriques

# ─────────────────────────────────────────
# 6. PRÉPARER DONNÉES JSON POUR HTML
# ─────────────────────────────────────────
def preparer_json(bt, step=3):
    """Sous-échantillonne pour alléger le HTML."""
    sub = bt.iloc[::step].copy()
    data = []
    for date, row in sub.iterrows():
        data.append({
            'date':       date.strftime('%Y-%m-%d'),
            'cum_bh':     round(float(row['cum_bh']), 4),
            'cum_strat':  round(float(row['cum_strat']), 4),
            'proba':      round(float(row.get('proba', 0.5)), 3),
            'signal':     str(row.get('signal', 'NEUTRE')),
            'regime':     str(row.get('regime', '')),
            'prix':       round(float(row['close']), 2),
            'dd_bh':      round(float(row['cum_bh'] / bt['cum_bh'].cummax()[row.name] - 1) * 100, 3)
                          if row.name in bt.index else 0,
            'dd_strat':   round(float(row['cum_strat'] / bt['cum_strat'].cummax()[row.name] - 1) * 100, 3)
                          if row.name in bt.index else 0,
        })
    return data

# ─────────────────────────────────────────
# 7. GÉNÉRATION HTML BACKTEST
# ─────────────────────────────────────────
def generer_html_backtest(metriques, bt_data, bt):
    m    = metriques
    gain = m['ret_total_strat'] - m['ret_total_bh']
    col  = '#00ff88' if gain >= 0 else '#ff3355'

    # Distribution mensuelle des signaux
    bt['mois'] = bt.index.to_period('M').astype(str)
    monthly = bt.groupby('mois').apply(lambda g: {
        'ret_strat': round(float((1+g['ret_strat']).prod()-1)*100, 2),
        'ret_bh':    round(float((1+g['ret_daily']).prod()-1)*100, 2),
        'n_achat':   int((g['signal']=='ACHAT').sum()),
        'n_vente':   int((g['signal']=='VENTE').sum()),
    }).to_dict()

    monthly_json = json.dumps([
        {'mois': k, **v} for k,v in list(monthly.items())[-36:]
    ])
    bt_json    = json.dumps(bt_data)
    met_json   = json.dumps(m)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Backtest — S&P 500 Regime Switching</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Archivo+Black&family=Archivo:wght@300;400;600&display=swap" rel="stylesheet"/>
<style>
:root{{
  --bg:#07090d; --bg2:#0e1117; --bg3:#151b24; --border:#1c2530;
  --green:#00e676; --red:#ff1744; --yellow:#ffd600; --blue:#2979ff;
  --purple:#d500f9; --cyan:#00e5ff;
  --text:#eceff4; --muted:#546e7a;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'Archivo',sans-serif;min-height:100vh}}
body::before{{content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse 60% 40% at 10% 10%,rgba(0,230,118,.05) 0,transparent 60%),
             radial-gradient(ellipse 50% 60% at 90% 90%,rgba(41,121,255,.05) 0,transparent 60%);
  pointer-events:none;z-index:0}}

/* HEADER */
.hdr{{position:relative;z-index:10;display:flex;align-items:center;justify-content:space-between;
  padding:18px 36px;border-bottom:1px solid var(--border);background:rgba(14,17,23,.9);
  backdrop-filter:blur(12px)}}
.hdr-logo{{font-family:'Archivo Black',sans-serif;font-size:16px;letter-spacing:3px;
  background:linear-gradient(135deg,var(--green),var(--cyan));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hdr-meta{{font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:1px}}

/* TABS */
.tabs{{position:relative;z-index:10;display:flex;padding:0 36px;
  background:var(--bg2);border-bottom:1px solid var(--border)}}
.tab{{padding:14px 24px;font-family:'DM Mono',monospace;font-size:10px;font-weight:500;
  letter-spacing:2px;text-transform:uppercase;color:var(--muted);cursor:pointer;
  border-bottom:2px solid transparent;transition:all .2s;user-select:none}}
.tab:hover{{color:var(--text)}}
.tab.on{{color:var(--green);border-bottom-color:var(--green)}}

/* PANELS */
.panel{{display:none;position:relative;z-index:5;animation:fi .3s ease}}
.panel.on{{display:block}}
@keyframes fi{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}

/* ── ONGLET 1 — RÉSUMÉ ── */
.page{{padding:32px 36px;max-width:1440px;margin:0 auto}}

.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
.kpi{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  padding:22px 24px;position:relative;overflow:hidden}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
.kpi.g::before{{background:var(--green)}}
.kpi.r::before{{background:var(--red)}}
.kpi.b::before{{background:var(--blue)}}
.kpi.y::before{{background:var(--yellow)}}
.kpi-label{{font-family:'DM Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
.kpi-val{{font-family:'Archivo Black',sans-serif;font-size:28px;line-height:1}}
.kpi-sub{{font-size:11px;color:var(--muted);margin-top:6px}}
.kpi-sub span{{font-weight:600}}

.compare-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
.cmp-card{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:24px}}
.cmp-title{{font-family:'DM Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;
  padding-bottom:10px;border-bottom:1px solid var(--border)}}
.cmp-row{{display:flex;justify-content:space-between;align-items:center;
  padding:9px 0;border-bottom:1px solid var(--border)20;font-size:12px}}
.cmp-row:last-child{{border-bottom:none}}
.cmp-key{{color:var(--muted)}}
.cmp-v{{font-family:'DM Mono',monospace;font-weight:500}}
.green{{color:var(--green)}} .red{{color:var(--red)}}
.yellow{{color:var(--yellow)}} .blue{{color:var(--blue)}}

.signals-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}}
.sig-box{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  padding:20px;text-align:center}}
.sig-ico{{font-size:32px;margin-bottom:8px}}
.sig-n{{font-family:'Archivo Black',sans-serif;font-size:32px}}
.sig-lbl{{font-size:10px;color:var(--muted);letter-spacing:1px;margin-top:4px}}

/* CHART AREA */
.chart-box{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  overflow:hidden;margin-bottom:20px}}
.chart-hdr{{display:flex;align-items:center;justify-content:space-between;
  padding:14px 20px;border-bottom:1px solid var(--border)}}
.chart-title{{font-family:'DM Mono',monospace;font-size:11px;font-weight:500;letter-spacing:1px}}
.legend{{display:flex;gap:16px}}
.leg{{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted)}}
.leg-dot{{width:24px;height:2px;border-radius:1px}}
canvas{{display:block;width:100%}}

/* ── ONGLET 2 — COURBES ── */
/* ── ONGLET 3 — MENSUEL ── */
.month-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}}
.month-card{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:14px}}
.mc-date{{font-family:'DM Mono',monospace;font-size:9px;color:var(--muted);margin-bottom:8px}}
.mc-ret{{font-family:'Archivo Black',sans-serif;font-size:20px}}
.mc-sub{{font-size:10px;color:var(--muted);margin-top:4px}}
.mc-sigs{{display:flex;gap:8px;margin-top:8px;font-size:10px}}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-logo">BACKTEST · S&P 500</div>
  <div class="hdr-meta">
    {m['date_debut']} → {m['date_fin']} · {m['n_jours']} jours
  </div>
</div>

<div class="tabs">
  <div class="tab on" onclick="sw('resume',this)">📋 Résumé</div>
  <div class="tab" onclick="sw('courbes',this)">📈 Courbes</div>
  <div class="tab" onclick="sw('mensuel',this)">📅 Mensuel</div>
</div>

<!-- ═══════════════════════════════ RÉSUMÉ ═══════════════════════════════ -->
<div id="p-resume" class="panel on">
<div class="page">

  <div class="kpi-grid">
    <div class="kpi {'g' if m['ret_total_strat'] >= 0 else 'r'}">
      <div class="kpi-label">Rendement Total Stratégie</div>
      <div class="kpi-val {'green' if m['ret_total_strat']>=0 else 'red'}">{m['ret_total_strat']:+.1f}%</div>
      <div class="kpi-sub">vs Buy&Hold <span class="{'green' if m['ret_total_bh']>=0 else 'red'}">{m['ret_total_bh']:+.1f}%</span></div>
    </div>
    <div class="kpi {'g' if m['sharpe_strat'] >= m['sharpe_bh'] else 'r'}">
      <div class="kpi-label">Sharpe Ratio Stratégie</div>
      <div class="kpi-val {'green' if m['sharpe_strat']>=m['sharpe_bh'] else 'red'}">{m['sharpe_strat']}</div>
      <div class="kpi-sub">vs Buy&Hold <span>{m['sharpe_bh']}</span></div>
    </div>
    <div class="kpi {'g' if abs(m['mdd_strat']) < abs(m['mdd_bh']) else 'r'}">
      <div class="kpi-label">Max Drawdown Stratégie</div>
      <div class="kpi-val {'green' if abs(m['mdd_strat'])<abs(m['mdd_bh']) else 'red'}">{m['mdd_strat']:.1f}%</div>
      <div class="kpi-sub">vs Buy&Hold <span class="red">{m['mdd_bh']:.1f}%</span></div>
    </div>
    <div class="kpi b">
      <div class="kpi-label">Win Rate</div>
      <div class="kpi-val blue">{m['win_rate']}%</div>
      <div class="kpi-sub">Investi <span>{m['jours_investis']}%</span> du temps</div>
    </div>
  </div>

  <div class="compare-grid">
    <div class="cmp-card">
      <div class="cmp-title">Stratégie Regime Switching</div>
      <div class="cmp-row"><span class="cmp-key">Rendement annualisé</span><span class="cmp-v {'green' if m['ret_ann_strat']>=0 else 'red'}">{m['ret_ann_strat']:+.2f}%</span></div>
      <div class="cmp-row"><span class="cmp-key">Sharpe</span><span class="cmp-v">{m['sharpe_strat']}</span></div>
      <div class="cmp-row"><span class="cmp-key">Sortino</span><span class="cmp-v">{m['sortino_strat']}</span></div>
      <div class="cmp-row"><span class="cmp-key">Calmar</span><span class="cmp-v">{m['calmar_strat']}</span></div>
      <div class="cmp-row"><span class="cmp-key">Max Drawdown</span><span class="cmp-v red">{m['mdd_strat']:.2f}%</span></div>
      <div class="cmp-row"><span class="cmp-key">Nb trades</span><span class="cmp-v">{m['n_trades']}</span></div>
      <div class="cmp-row"><span class="cmp-key">Win rate</span><span class="cmp-v green">{m['win_rate']}%</span></div>
    </div>
    <div class="cmp-card">
      <div class="cmp-title">Buy & Hold (référence)</div>
      <div class="cmp-row"><span class="cmp-key">Rendement annualisé</span><span class="cmp-v {'green' if m['ret_ann_bh']>=0 else 'red'}">{m['ret_ann_bh']:+.2f}%</span></div>
      <div class="cmp-row"><span class="cmp-key">Sharpe</span><span class="cmp-v">{m['sharpe_bh']}</span></div>
      <div class="cmp-row"><span class="cmp-key">Sortino</span><span class="cmp-v">{m['sortino_bh']}</span></div>
      <div class="cmp-row"><span class="cmp-key">Calmar</span><span class="cmp-v">{m['calmar_bh']}</span></div>
      <div class="cmp-row"><span class="cmp-key">Max Drawdown</span><span class="cmp-v red">{m['mdd_bh']:.2f}%</span></div>
      <div class="cmp-row"><span class="cmp-key">Nb trades</span><span class="cmp-v">1</span></div>
      <div class="cmp-row"><span class="cmp-key">Investi</span><span class="cmp-v">100% du temps</span></div>
    </div>
  </div>

  <div class="signals-row">
    <div class="sig-box">
      <div class="sig-ico">🟢</div>
      <div class="sig-n green">{m['n_achat']}</div>
      <div class="sig-lbl">Signaux ACHAT</div>
    </div>
    <div class="sig-box">
      <div class="sig-ico">🟡</div>
      <div class="sig-n yellow">{m['n_neutre']}</div>
      <div class="sig-lbl">Signaux NEUTRE</div>
    </div>
    <div class="sig-box">
      <div class="sig-ico">🔴</div>
      <div class="sig-n red">{m['n_vente']}</div>
      <div class="sig-lbl">Signaux VENTE</div>
    </div>
  </div>

  <div class="chart-box">
    <div class="chart-hdr">
      <span class="chart-title">PERFORMANCE CUMULÉE</span>
      <div class="legend">
        <div class="leg"><div class="leg-dot" style="background:#00e676"></div>Stratégie</div>
        <div class="leg"><div class="leg-dot" style="background:#2979ff"></div>Buy & Hold</div>
      </div>
    </div>
    <canvas id="cumChart" height="280"></canvas>
  </div>

  <div class="chart-box">
    <div class="chart-hdr">
      <span class="chart-title">DRAWDOWN</span>
      <div class="legend">
        <div class="leg"><div class="leg-dot" style="background:#00e676"></div>Stratégie</div>
        <div class="leg"><div class="leg-dot" style="background:#2979ff"></div>Buy & Hold</div>
      </div>
    </div>
    <canvas id="ddChart" height="160"></canvas>
  </div>

</div>
</div>

<!-- ═══════════════════════════════ COURBES ═══════════════════════════════ -->
<div id="p-courbes" class="panel">
<div class="page">
  <div class="chart-box">
    <div class="chart-hdr">
      <span class="chart-title">PROBABILITÉ DE HAUSSE — SIGNAL JOURNALIER</span>
      <div class="legend">
        <div class="leg"><div class="leg-dot" style="background:#00e676"></div>ACHAT</div>
        <div class="leg"><div class="leg-dot" style="background:#ffd600"></div>NEUTRE</div>
        <div class="leg"><div class="leg-dot" style="background:#ff1744"></div>VENTE</div>
      </div>
    </div>
    <canvas id="probaChart" height="240"></canvas>
  </div>
  <div class="chart-box">
    <div class="chart-hdr"><span class="chart-title">PRIX S&P 500</span></div>
    <canvas id="prixChart" height="220"></canvas>
  </div>
</div>
</div>

<!-- ═══════════════════════════════ MENSUEL ═══════════════════════════════ -->
<div id="p-mensuel" class="panel">
<div class="page">
  <div class="month-grid" id="monthGrid"></div>
</div>
</div>

<script>
const BT   = {bt_json};
const MET  = {met_json};
const MON  = {monthly_json};

function sw(name,el){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  document.getElementById('p-'+name).classList.add('on');
  el.classList.add('on');
  if(name==='resume') {{drawCum(); drawDD();}}
  if(name==='courbes'){{drawProba(); drawPrix();}}
  if(name==='mensuel') drawMonthly();
}}

function getCtx(id,h){{
  const c=document.getElementById(id);
  c.width=c.parentElement.clientWidth;
  c.height=h||280;
  const ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  return{{ctx,w:c.width,h:c.height}};
}}
function mapY(v,mn,mx,t,b){{return t+(1-(v-mn)/(mx-mn))*(b-t);}}

function drawLine(ctx,data,key,color,w,h,PAD,mn,mx,lw=1.8){{
  ctx.beginPath();let f=true;
  data.forEach((d,i)=>{{
    const x=PAD.l+(i/(data.length-1))*(w-PAD.l-PAD.r);
    const y=mapY(d[key],mn,mx,PAD.t,h-PAD.b);
    f?ctx.moveTo(x,y):ctx.lineTo(x,y);f=false;
  }});
  ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.stroke();
}}

function drawGrid(ctx,w,h,PAD,mn,mx,steps=5,unit=''){{
  ctx.fillStyle='#07090d';ctx.fillRect(0,0,w,h);
  for(let i=0;i<=steps;i++){{
    const v=mn+(mx-mn)*i/steps;
    const y=mapY(v,mn,mx,PAD.t,h-PAD.b);
    ctx.strokeStyle='#1c2530';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(PAD.l,y);ctx.lineTo(w-PAD.r,y);ctx.stroke();
    ctx.fillStyle='#546e7a';ctx.font='9px DM Mono';
    ctx.fillText(v.toFixed(1)+unit,4,y+3);
  }}
}}

// ── Cumul ──────────────────────────────────────────────────
function drawCum(){{
  const{{ctx,w,h}}=getCtx('cumChart',280);
  const PAD={{l:50,r:20,t:16,b:30}};
  const mn=Math.min(...BT.map(d=>Math.min(d.cum_bh,d.cum_strat)))*0.98;
  const mx=Math.max(...BT.map(d=>Math.max(d.cum_bh,d.cum_strat)))*1.02;
  drawGrid(ctx,w,h,PAD,mn,mx,6,'x');
  drawLine(ctx,BT,'cum_bh',  '#2979ff',w,h,PAD,mn,mx,1.5);
  drawLine(ctx,BT,'cum_strat','#00e676',w,h,PAD,mn,mx,2);
  // Dates
  const step=Math.ceil(BT.length/8);
  ctx.fillStyle='#546e7a';ctx.font='9px DM Mono';
  BT.forEach((d,i)=>{{if(i%step===0){{
    const x=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    ctx.fillText(d.date.slice(0,7),x-18,h-6);
  }}}});
}}

// ── Drawdown ───────────────────────────────────────────────
function drawDD(){{
  const{{ctx,w,h}}=getCtx('ddChart',160);
  const PAD={{l:50,r:20,t:10,b:26}};
  const mn=Math.min(...BT.map(d=>Math.min(d.dd_bh||0,d.dd_strat||0)))*1.05;
  const mx=0;
  drawGrid(ctx,w,h,PAD,mn,mx,4,'%');
  // Fill BH
  ctx.beginPath();
  BT.forEach((d,i)=>{{const x=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    i===0?ctx.moveTo(x,mapY(0,mn,mx,PAD.t,h-PAD.b)):ctx.lineTo(x,mapY(d.dd_bh||0,mn,mx,PAD.t,h-PAD.b));
  }});
  ctx.fillStyle='rgba(41,121,255,0.15)';ctx.fill();
  drawLine(ctx,BT,'dd_bh','#2979ff',w,h,PAD,mn,mx,1.2);
  drawLine(ctx,BT,'dd_strat','#00e676',w,h,PAD,mn,mx,1.5);
}}

// ── Proba ─────────────────────────────────────────────────
function drawProba(){{
  const{{ctx,w,h}}=getCtx('probaChart',240);
  const PAD={{l:50,r:20,t:12,b:30}};
  drawGrid(ctx,w,h,PAD,0,1,4,'');
  // Barres colorées
  const bw=Math.max(1,(w-PAD.l-PAD.r)/BT.length);
  BT.forEach((d,i)=>{{
    const x=PAD.l+i*bw;
    const col=d.signal==='ACHAT'?'#00e67666':d.signal==='VENTE'?'#ff174466':'#ffd60044';
    ctx.fillStyle=col;
    const y=mapY(d.proba,0,1,PAD.t,h-PAD.b);
    ctx.fillRect(x,y,bw,h-PAD.b-y);
  }});
  // Seuils
  [0.6,0.5,0.4].forEach(v=>{{
    ctx.strokeStyle=v===0.5?'#ffffff22':'#ffffff44';
    ctx.lineWidth=1;ctx.setLineDash(v===0.5?[3,3]:[]);
    ctx.beginPath();ctx.moveTo(PAD.l,mapY(v,0,1,PAD.t,h-PAD.b));
    ctx.lineTo(w-PAD.r,mapY(v,0,1,PAD.t,h-PAD.b));ctx.stroke();
    ctx.setLineDash([]);
  }});
  drawLine(ctx,BT,'proba','#eceff4',w,h,PAD,0,1,1.2);
  // Dates
  const step=Math.ceil(BT.length/8);
  ctx.fillStyle='#546e7a';ctx.font='9px DM Mono';
  BT.forEach((d,i)=>{{if(i%step===0){{
    const x=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    ctx.fillText(d.date.slice(0,7),x-18,h-6);
  }}}});
}}

// ── Prix ──────────────────────────────────────────────────
function drawPrix(){{
  const{{ctx,w,h}}=getCtx('prixChart',220);
  const PAD={{l:60,r:20,t:12,b:30}};
  const mn=Math.min(...BT.map(d=>d.prix))*0.99;
  const mx=Math.max(...BT.map(d=>d.prix))*1.01;
  drawGrid(ctx,w,h,PAD,mn,mx,5,'');
  // Zones signal
  BT.forEach((d,i)=>{{
    if(i===BT.length-1)return;
    const x1=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    const x2=PAD.l+((i+1)/(BT.length-1))*(w-PAD.l-PAD.r);
    const col=d.signal==='ACHAT'?'rgba(0,230,118,.08)':d.signal==='VENTE'?'rgba(255,23,68,.08)':null;
    if(col){{ctx.fillStyle=col;ctx.fillRect(x1,PAD.t,x2-x1,h-PAD.t-PAD.b);}}
  }});
  drawLine(ctx,BT,'prix','#eceff4',w,h,PAD,mn,mx,1.5);
  const step=Math.ceil(BT.length/8);
  ctx.fillStyle='#546e7a';ctx.font='9px DM Mono';
  BT.forEach((d,i)=>{{if(i%step===0){{
    const x=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    ctx.fillText(d.date.slice(0,7),x-18,h-6);
  }}}});
  // Prix
  ctx.fillStyle='#546e7a';ctx.textAlign='right';
  BT.forEach((d,i)=>{{if(i%Math.ceil(BT.length/5)===0){{
    const y=mapY(d.prix,mn,mx,PAD.t,h-PAD.b);
    ctx.fillText(d.prix.toFixed(0),PAD.l-4,y+3);
  }}}});
  ctx.textAlign='left';
}}

// ── Mensuel ───────────────────────────────────────────────
function drawMonthly(){{
  const grid=document.getElementById('monthGrid');
  grid.innerHTML='';
  MON.forEach(m=>{{
    const pos=m.ret_strat>=0;
    const col=pos?'#00e676':'#ff1744';
    grid.innerHTML+=`<div class="month-card">
      <div class="mc-date">${{m.mois}}</div>
      <div class="mc-ret" style="color:${{col}}">${{m.ret_strat>=0?'+':''}}${{m.ret_strat}}%</div>
      <div class="mc-sub">BH : ${{m.ret_bh>=0?'+':''}}${{m.ret_bh}}%</div>
      <div class="mc-sigs">
        <span style="color:#00e676">▲${{m.n_achat}}</span>
        <span style="color:#ff1744">▼${{m.n_vente}}</span>
      </div>
    </div>`;
  }});
}}

// Init
window.addEventListener('load',()=>{{drawCum();drawDD();}});
window.addEventListener('resize',()=>{{drawCum();drawDD();}});
</script>
</body>
</html>"""
    return html

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  BACKTEST — REGIME SWITCHING S&P 500")
    print("=" * 55)

    print("\nChargement des données...")
    df = charger_donnees()
    print(f"  {len(df)} jours ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")

    print("\nChargement des modèles...")
    modeles, params = charger_modeles()
    print(f"  {len(modeles)} modèles chargés : {list(modeles.keys())}")

    if not modeles:
        print("  ⚠️  Lance d'abord : python 2_train_models_FINAL_V3.py")
        exit()

    print("\nPrédictions sur tout l'historique...")
    df = predire_historique(df, modeles, params)

    print("\nSimulation backtest (frais 0.1%)...")
    bt = simuler_backtest(df, frais=0.001)

    print("\nCalcul des métriques...")
    metriques = calculer_metriques(bt)

    print("\n" + "=" * 55)
    print("  RÉSULTATS")
    print("=" * 55)
    print(f"  Rendement Stratégie : {metriques['ret_total_strat']:+.1f}%")
    print(f"  Rendement Buy&Hold  : {metriques['ret_total_bh']:+.1f}%")
    print(f"  Sharpe Stratégie    : {metriques['sharpe_strat']}")
    print(f"  Sharpe Buy&Hold     : {metriques['sharpe_bh']}")
    print(f"  Max DD Stratégie    : {metriques['mdd_strat']:.1f}%")
    print(f"  Max DD Buy&Hold     : {metriques['mdd_bh']:.1f}%")
    print(f"  Win Rate            : {metriques['win_rate']}%")

    print("\nGénération du rapport HTML...")
    # Calcul drawdowns pour JSON
    bt['dd_bh']    = (bt['cum_bh']    / bt['cum_bh'].cummax()    - 1) * 100
    bt['dd_strat'] = (bt['cum_strat'] / bt['cum_strat'].cummax() - 1) * 100
    bt_data   = preparer_json(bt, step=2)

    html = generer_html_backtest(metriques, bt_data, bt)

    chemin = "outputs/backtest.html"
    with open(chemin, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n  ✓ Rapport sauvegardé → {chemin}")
    print(f"  Lance : start outputs\\backtest.html")
    os.system(f'start "" "{os.path.abspath(chemin)}"')