# ============================================================
#  7_backtest_long_only.py
#  Stratégie : Long Only sur signaux de FORTE HAUSSE
#  Règle : P(hausse) > seuil → BUY | Sinon → CASH
#  Compare vs Buy & Hold
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
    df   = pd.read_sql("SELECT * FROM sp500_extreme_features",
                        conn, index_col='date', parse_dates=['date'])
    conn.close()
    df.sort_index(inplace=True)
    return df

def charger_modele_hausse():
    try:
        mod    = joblib.load("models/modele_extreme_hausse.pkl")
        params = joblib.load("models/extreme_params.pkl")
        print(f"  ✓ Modèle hausse chargé (AUC={mod.get('auc',0):.4f})")
        return mod, params
    except Exception as e:
        print(f"  ⚠️  Modèle non trouvé : {e}")
        print("  Lance d'abord : python 2_train_extreme.py")
        exit()

# ─────────────────────────────────────────
# 2. PRÉDICTIONS SUR TOUT L'HISTORIQUE
# ─────────────────────────────────────────
def predire_historique(df, mod):
    feats  = mod['feats']
    cols   = [f for f in feats if f in df.columns]
    X      = df[cols].copy()

    # Nettoyage
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
        q1, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col]  = X[col].clip(q1, q99)

    X_s    = np.nan_to_num(mod['scaler'].transform(X.values))
    p_xgb  = mod['xgb'].predict_proba(X_s)[:, 1]
    p_rf   = mod['rf'].predict_proba(X_s)[:, 1]
    probas = mod['w_xgb'] * p_xgb + mod['w_rf'] * p_rf

    print(f"  Proba min={probas.min():.3f} | max={probas.max():.3f} | moy={probas.mean():.3f}")
    return probas

# ─────────────────────────────────────────
# 3. BACKTEST MULTI-SEUILS
# ─────────────────────────────────────────
def position_avec_holding(probas, seuil, holding=10):
    pos = np.zeros(len(probas))
    jours_restants = 0
    for i in range(1, len(probas)):
        if probas[i-1] >= seuil:
            jours_restants = holding
        if jours_restants > 0:
            pos[i] = 1
            jours_restants -= 1
    return pos

def backtest_seuil(df, probas, seuil, frais=0.001, holding=10):
    bt = df[['close']].copy()
    bt['ret_daily'] = bt['close'].pct_change()
    bt['proba']     = probas

    bt['position']   = position_avec_holding(probas, seuil, holding)
    bt['pos_change'] = bt['position'].diff().abs().fillna(0)
    bt['frais_app']  = bt['pos_change'] * frais
    bt['ret_strat']  = bt['position'] * bt['ret_daily'] - bt['frais_app']

    bt.dropna(inplace=True)
    bt['cum_bh']    = (1 + bt['ret_daily']).cumprod()
    bt['cum_strat'] = (1 + bt['ret_strat']).cumprod()

    return bt

def calculer_metriques(bt):
    rf_j    = 0.02 / 252
    n       = len(bt)
    ann     = 252 / n

    def sharpe(rets):
        ex = rets - rf_j
        return (ex.mean() / ex.std()) * np.sqrt(252) if ex.std() > 0 else 0

    def max_dd(cum):
        return float((cum / cum.cummax() - 1).min() * 100)

    def sortino(rets):
        ex   = rets - rf_j
        neg  = ex[ex < 0]
        down = neg.std() * np.sqrt(252) if len(neg) > 0 else 1
        return ex.mean() * 252 / down

    ret_strat   = float(bt['cum_strat'].iloc[-1] - 1)
    ret_bh      = float(bt['cum_bh'].iloc[-1] - 1)
    ann_strat   = float((bt['cum_strat'].iloc[-1] ** ann) - 1)
    ann_bh      = float((bt['cum_bh'].iloc[-1] ** ann) - 1)

    return {
        'ret_total_strat':  round(ret_strat  * 100, 2),
        'ret_total_bh':     round(ret_bh     * 100, 2),
        'ret_ann_strat':    round(ann_strat  * 100, 2),
        'ret_ann_bh':       round(ann_bh     * 100, 2),
        'sharpe_strat':     round(sharpe(bt['ret_strat']),   3),
        'sharpe_bh':        round(sharpe(bt['ret_daily']),   3),
        'sortino_strat':    round(sortino(bt['ret_strat']),  3),
        'sortino_bh':       round(sortino(bt['ret_daily']),  3),
        'mdd_strat':        round(max_dd(bt['cum_strat']),   2),
        'mdd_bh':           round(max_dd(bt['cum_bh']),      2),
        'win_rate':         round(float((bt['ret_strat']>0).mean()) * 100, 2),
        'jours_investis':   round(float((bt['position']>0).mean()) * 100, 2),
        'n_trades':         int(bt['pos_change'].sum()),
        'n_signaux':        int((bt['proba'] >= 0.55).sum()),
        'date_debut':       bt.index[0].strftime('%Y-%m-%d'),
        'date_fin':         bt.index[-1].strftime('%Y-%m-%d'),
        'n_jours':          n,
    }

# ─────────────────────────────────────────
# 4. RÉSUMÉ CONSOLE
# ─────────────────────────────────────────
def afficher_resultats(seuil, m):
    print(f"\n  Seuil {seuil:.2f} :")
    print(f"    Ret strat     : {m['ret_total_strat']:+.1f}%  vs BH : {m['ret_total_bh']:+.1f}%")
    print(f"    Ret ann.      : {m['ret_ann_strat']:+.2f}%  vs BH : {m['ret_ann_bh']:+.2f}%")
    print(f"    Sharpe        : {m['sharpe_strat']:.3f}  vs BH : {m['sharpe_bh']:.3f}")
    print(f"    Sortino       : {m['sortino_strat']:.3f}  vs BH : {m['sortino_bh']:.3f}")
    print(f"    Max DD        : {m['mdd_strat']:.1f}%  vs BH : {m['mdd_bh']:.1f}%")
    print(f"    Win Rate      : {m['win_rate']}%")
    print(f"    Jours investis: {m['jours_investis']}%")
    print(f"    Nb trades     : {m['n_trades']}")

# ─────────────────────────────────────────
# 5. GÉNÉRATION HTML
# ─────────────────────────────────────────
def generer_html(resultats_seuils, bt_best, m_best, seuil_best, probas, df):
    # Données JSON pour les graphiques
    step = max(1, len(bt_best) // 500)
    sub  = bt_best.iloc[::step]

    bt_json = json.dumps([{
        'date':      d.strftime('%Y-%m-%d'),
        'cum_bh':    round(float(r['cum_bh']),    4),
        'cum_strat': round(float(r['cum_strat']), 4),
        'proba':     round(float(r['proba']),      3),
        'pos':       int(r['position']),
        'dd_bh':     round(float(r['cum_bh']    / bt_best['cum_bh'].cummax()[r.name]    - 1) * 100, 2),
        'dd_strat':  round(float(r['cum_strat'] / bt_best['cum_strat'].cummax()[r.name] - 1) * 100, 2),
        'prix':      round(float(r['close']), 2),
    } for d, r in sub.iterrows()])

    seuils_json = json.dumps([{
        'seuil':   s,
        'sharpe':  m['sharpe_strat'],
        'ret':     m['ret_ann_strat'],
        'mdd':     m['mdd_strat'],
        'investi': m['jours_investis'],
    } for s, m in resultats_seuils.items()])

    m = m_best

    # Distribution des probas
    hist_data = np.histogram(probas, bins=20, range=(0,1))
    hist_json = json.dumps({
        'counts': hist_data[0].tolist(),
        'edges':  [round(e,2) for e in hist_data[1].tolist()],
    })

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Backtest Long Only — S&P 500</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;700&family=Manrope:wght@300;400;600;800;900&display=swap" rel="stylesheet"/>
<style>
:root{{
  --bg:#05080c;--bg2:#0a0f16;--bg3:#0f1822;--border:#162030;
  --green:#00ff9d;--red:#ff2d5b;--yellow:#ffd000;--blue:#0090ff;
  --cyan:#00d4ff;--purple:#b36eff;
  --text:#dde6f0;--muted:#3d5a73;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'Manrope',sans-serif;min-height:100vh}}
body::before{{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 80% 50% at 0% 0%,rgba(0,255,157,.04) 0,transparent 60%),
             radial-gradient(ellipse 60% 80% at 100% 100%,rgba(0,144,255,.04) 0,transparent 60%)}}

.hdr{{position:relative;z-index:10;padding:20px 40px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:rgba(10,15,22,.85);backdrop-filter:blur(16px)}}
.logo{{font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:700;letter-spacing:3px;
  background:linear-gradient(90deg,var(--green),var(--cyan));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hdr-r{{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:1px}}

.tabs{{display:flex;padding:0 40px;background:var(--bg2);border-bottom:1px solid var(--border);
  position:relative;z-index:10}}
.tab{{padding:14px 22px;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:500;
  letter-spacing:2px;text-transform:uppercase;color:var(--muted);cursor:pointer;
  border-bottom:2px solid transparent;transition:all .2s;user-select:none}}
.tab:hover{{color:var(--text)}}
.tab.on{{color:var(--green);border-bottom-color:var(--green)}}
.panel{{display:none;position:relative;z-index:5;animation:fi .3s ease}}
.panel.on{{display:block}}
@keyframes fi{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}

.page{{padding:32px 40px;max-width:1400px;margin:0 auto}}

/* KPIs */
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
.kpi{{background:var(--bg2);border:1px solid var(--border);border-radius:14px;
  padding:24px;position:relative;overflow:hidden}}
.kpi::after{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
.kpi.g::after{{background:linear-gradient(90deg,var(--green),var(--cyan))}}
.kpi.b::after{{background:linear-gradient(90deg,var(--blue),var(--purple))}}
.kpi.y::after{{background:var(--yellow)}}
.kpi.r::after{{background:var(--red)}}
.kpi-lbl{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}}
.kpi-val{{font-family:'Manrope',sans-serif;font-size:30px;font-weight:900;line-height:1}}
.kpi-sub{{font-size:11px;color:var(--muted);margin-top:8px}}
.kpi-sub b{{color:var(--text)}}
.green{{color:var(--green)}} .red{{color:var(--red)}}
.blue{{color:var(--blue)}}   .yellow{{color:var(--yellow)}}
.cyan{{color:var(--cyan)}}

/* Compare */
.cmp-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
.cmp{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:24px}}
.cmp-ttl{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;
  padding-bottom:10px;border-bottom:1px solid var(--border)}}
.row{{display:flex;justify-content:space-between;align-items:center;
  padding:9px 0;border-bottom:1px solid rgba(22,32,48,.6);font-size:12px}}
.row:last-child{{border:none}}
.rk{{color:var(--muted)}}
.rv{{font-family:'IBM Plex Mono',monospace;font-weight:500}}

/* Charts */
.chart-box{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  overflow:hidden;margin-bottom:20px}}
.chart-hdr{{display:flex;align-items:center;justify-content:space-between;
  padding:14px 20px;border-bottom:1px solid var(--border)}}
.chart-ttl{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:500;letter-spacing:1px}}
.leg{{display:flex;gap:16px}}
.li{{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted)}}
.ld{{width:24px;height:2px;border-radius:1px}}
canvas{{display:block;width:100%}}

/* Seuil selector */
.seuil-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:24px}}
.seuil-btn{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;
  padding:14px 10px;text-align:center;cursor:pointer;transition:all .2s}}
.seuil-btn:hover,.seuil-btn.active{{border-color:var(--green);background:rgba(0,255,157,.06)}}
.sb-val{{font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:700;color:var(--green)}}
.sb-sub{{font-size:10px;color:var(--muted);margin-top:4px}}

/* Règle */
.regle{{background:var(--bg2);border:1px solid var(--green)33;border-radius:12px;
  padding:24px;margin-bottom:24px;position:relative;overflow:hidden}}
.regle::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 60% 80% at 0% 50%,rgba(0,255,157,.06) 0,transparent 70%);
  pointer-events:none}}
.regle-ttl{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--green);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:16px}}
.regle-rule{{font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:700;
  color:var(--text);line-height:2.2}}
.regle-rule .g{{color:var(--green)}}
.regle-rule .r{{color:var(--red)}}
.regle-rule .y{{color:var(--yellow)}}

/* Signal today */
.signal-box{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  padding:24px;text-align:center}}
</style>
</head>
<body>

<div class="hdr">
  <div class="logo">BACKTEST · LONG ONLY · S&P 500</div>
  <div class="hdr-r">{m['date_debut']} → {m['date_fin']} · {m['n_jours']}j · seuil optimal : {seuil_best:.2f}</div>
</div>

<div class="tabs">
  <div class="tab on"  onclick="sw('resume',this)">📋 Résumé</div>
  <div class="tab"     onclick="sw('courbes',this)">📈 Courbes</div>
  <div class="tab"     onclick="sw('seuils',this)">🎯 Seuils</div>
</div>

<!-- ══════════ RÉSUMÉ ══════════ -->
<div id="p-resume" class="panel on">
<div class="page">

  <!-- Règle de trading -->
  <div class="regle">
    <div class="regle-ttl">Règle de trading</div>
    <div class="regle-rule">
      SI <span class="g">P(forte hausse &gt; +1.5% / 5j) &gt; {seuil_best:.2f}</span> → <span class="g">🟢 BUY</span><br>
      SINON → <span class="y">🟡 CASH</span> (aucune position)<br>
      <span class="r">❌ Jamais de short</span>
    </div>
  </div>

  <!-- KPIs -->
  <div class="kpi-grid">
    <div class="kpi {'g' if m['ret_total_strat'] >= m['ret_total_bh'] else 'r'}">
      <div class="kpi-lbl">Rendement Total</div>
      <div class="kpi-val {'green' if m['ret_total_strat']>=0 else 'red'}">{m['ret_total_strat']:+.1f}%</div>
      <div class="kpi-sub">vs B&H <b class="{'green' if m['ret_total_bh']>=0 else 'red'}">{m['ret_total_bh']:+.1f}%</b></div>
    </div>
    <div class="kpi {'g' if m['sharpe_strat'] >= m['sharpe_bh'] else 'b'}">
      <div class="kpi-lbl">Sharpe Ratio</div>
      <div class="kpi-val {'green' if m['sharpe_strat']>=m['sharpe_bh'] else 'blue'}">{m['sharpe_strat']}</div>
      <div class="kpi-sub">vs B&H <b>{m['sharpe_bh']}</b></div>
    </div>
    <div class="kpi {'g' if abs(m['mdd_strat']) < abs(m['mdd_bh']) else 'r'}">
      <div class="kpi-lbl">Max Drawdown</div>
      <div class="kpi-val red">{m['mdd_strat']:.1f}%</div>
      <div class="kpi-sub">vs B&H <b class="red">{m['mdd_bh']:.1f}%</b></div>
    </div>
    <div class="kpi b">
      <div class="kpi-lbl">Jours Investis</div>
      <div class="kpi-val blue">{m['jours_investis']}%</div>
      <div class="kpi-sub">Win rate <b>{m['win_rate']}%</b></div>
    </div>
  </div>

  <!-- Comparaison -->
  <div class="cmp-grid">
    <div class="cmp">
      <div class="cmp-ttl">🟢 Stratégie Long Only (seuil {seuil_best:.2f})</div>
      <div class="row"><span class="rk">Rendement annualisé</span><span class="rv {'green' if m['ret_ann_strat']>=0 else 'red'}">{m['ret_ann_strat']:+.2f}%</span></div>
      <div class="row"><span class="rk">Sharpe</span><span class="rv">{m['sharpe_strat']}</span></div>
      <div class="row"><span class="rk">Sortino</span><span class="rv">{m['sortino_strat']}</span></div>
      <div class="row"><span class="rk">Max Drawdown</span><span class="rv red">{m['mdd_strat']:.2f}%</span></div>
      <div class="row"><span class="rk">Win Rate</span><span class="rv green">{m['win_rate']}%</span></div>
      <div class="row"><span class="rk">Nb trades</span><span class="rv">{m['n_trades']}</span></div>
      <div class="row"><span class="rk">Temps en marché</span><span class="rv">{m['jours_investis']}%</span></div>
    </div>
    <div class="cmp">
      <div class="cmp-ttl">📊 Buy & Hold (référence)</div>
      <div class="row"><span class="rk">Rendement annualisé</span><span class="rv {'green' if m['ret_ann_bh']>=0 else 'red'}">{m['ret_ann_bh']:+.2f}%</span></div>
      <div class="row"><span class="rk">Sharpe</span><span class="rv">{m['sharpe_bh']}</span></div>
      <div class="row"><span class="rk">Sortino</span><span class="rv">{m['sortino_bh']}</span></div>
      <div class="row"><span class="rk">Max Drawdown</span><span class="rv red">{m['mdd_bh']:.2f}%</span></div>
      <div class="row"><span class="rk">Win Rate</span><span class="rv">{float((bt_best['ret_daily']>0).mean()*100):.1f}%</span></div>
      <div class="row"><span class="rk">Nb trades</span><span class="rv">1</span></div>
      <div class="row"><span class="rk">Temps en marché</span><span class="rv">100%</span></div>
    </div>
  </div>

  <!-- Chart perf -->
  <div class="chart-box">
    <div class="chart-hdr">
      <span class="chart-ttl">PERFORMANCE CUMULÉE</span>
      <div class="leg">
        <div class="li"><div class="ld" style="background:var(--green)"></div>Long Only</div>
        <div class="li"><div class="ld" style="background:var(--blue)"></div>Buy & Hold</div>
      </div>
    </div>
    <canvas id="cumChart" height="280"></canvas>
  </div>

  <!-- Chart DD -->
  <div class="chart-box">
    <div class="chart-hdr">
      <span class="chart-ttl">DRAWDOWN</span>
      <div class="leg">
        <div class="li"><div class="ld" style="background:var(--green)"></div>Stratégie</div>
        <div class="li"><div class="ld" style="background:var(--blue)"></div>Buy & Hold</div>
      </div>
    </div>
    <canvas id="ddChart" height="160"></canvas>
  </div>

</div>
</div>

<!-- ══════════ COURBES ══════════ -->
<div id="p-courbes" class="panel">
<div class="page">
  <div class="chart-box">
    <div class="chart-hdr"><span class="chart-ttl">PROBABILITÉ DE FORTE HAUSSE + ZONES D'ACHAT</span></div>
    <canvas id="probaChart" height="260"></canvas>
  </div>
  <div class="chart-box">
    <div class="chart-hdr"><span class="chart-ttl">DISTRIBUTION DES PROBABILITÉS</span></div>
    <canvas id="histChart" height="200"></canvas>
  </div>
</div>
</div>

<!-- ══════════ SEUILS ══════════ -->
<div id="p-seuils" class="panel">
<div class="page">
  <p style="color:var(--muted);font-size:12px;margin-bottom:16px">
    Cliquez sur un seuil pour voir ses métriques détaillées.
  </p>
  <div class="seuil-grid" id="seuilGrid"></div>
  <div id="seuilDetail"></div>
  <div class="chart-box" style="margin-top:20px">
    <div class="chart-hdr"><span class="chart-ttl">SHARPE RATIO PAR SEUIL</span></div>
    <canvas id="sharpeChart" height="200"></canvas>
  </div>
</div>
</div>

<script>
const BT      = {bt_json};
const SEUILS  = {seuils_json};
const HIST    = {hist_json};
const MBEST   = {json.dumps(m_best)};
const SBEST   = {seuil_best};

function sw(n,el){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  document.getElementById('p-'+n).classList.add('on'); el.classList.add('on');
  if(n==='resume') {{drawCum();drawDD();}}
  if(n==='courbes'){{drawProba();drawHist();}}
  if(n==='seuils') {{drawSeuils();drawSharpe();}}
}}

function getCtx(id,h){{
  const c=document.getElementById(id);
  c.width=c.parentElement.clientWidth; c.height=h||280;
  const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
  return{{ctx,w:c.width,h:c.height}};
}}
function mY(v,mn,mx,t,b){{return t+(1-(v-mn)/(mx-mn))*(b-t);}}

function drawGrid(ctx,w,h,PAD,mn,mx,steps,unit){{
  ctx.fillStyle='#0a0f16'; ctx.fillRect(0,0,w,h);
  for(let i=0;i<=steps;i++){{
    const v=mn+(mx-mn)*i/steps;
    const y=mY(v,mn,mx,PAD.t,h-PAD.b);
    ctx.strokeStyle='#162030'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(w-PAD.r,y); ctx.stroke();
    ctx.fillStyle='#3d5a73'; ctx.font='9px IBM Plex Mono';
    ctx.fillText(v.toFixed(1)+unit, 4, y+3);
  }}
}}

function drawLine(ctx,data,key,col,w,h,PAD,mn,mx,lw){{
  ctx.beginPath(); let f=true;
  data.forEach((d,i)=>{{
    const x=PAD.l+(i/(data.length-1))*(w-PAD.l-PAD.r);
    const y=mY(d[key],mn,mx,PAD.t,h-PAD.b);
    f?ctx.moveTo(x,y):ctx.lineTo(x,y); f=false;
  }});
  ctx.strokeStyle=col; ctx.lineWidth=lw||1.8; ctx.stroke();
}}

function drawCum(){{
  const{{ctx,w,h}}=getCtx('cumChart',280);
  const PAD={{l:50,r:16,t:16,b:30}};
  const mn=Math.min(...BT.map(d=>Math.min(d.cum_bh,d.cum_strat)))*0.98;
  const mx=Math.max(...BT.map(d=>Math.max(d.cum_bh,d.cum_strat)))*1.02;
  drawGrid(ctx,w,h,PAD,mn,mx,6,'x');
  // Zone investie
  BT.forEach((d,i)=>{{
    if(!d.pos||i>=BT.length-1)return;
    const x1=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    const x2=PAD.l+((i+1)/(BT.length-1))*(w-PAD.l-PAD.r);
    ctx.fillStyle='rgba(0,255,157,.06)';
    ctx.fillRect(x1,PAD.t,x2-x1,h-PAD.t-PAD.b);
  }});
  drawLine(ctx,BT,'cum_bh','#0090ff',w,h,PAD,mn,mx,1.5);
  drawLine(ctx,BT,'cum_strat','#00ff9d',w,h,PAD,mn,mx,2.2);
  // Dates
  const step=Math.ceil(BT.length/8);
  ctx.fillStyle='#3d5a73'; ctx.font='9px IBM Plex Mono';
  BT.forEach((d,i)=>{{if(i%step===0){{
    const x=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    ctx.fillText(d.date.slice(0,7),x-18,h-6);
  }}}});
}}

function drawDD(){{
  const{{ctx,w,h}}=getCtx('ddChart',160);
  const PAD={{l:50,r:16,t:10,b:26}};
  const mn=Math.min(...BT.map(d=>Math.min(d.dd_bh||0,d.dd_strat||0)))*1.1;
  const mx=0;
  drawGrid(ctx,w,h,PAD,mn,mx,4,'%');
  const y0=mY(0,mn,mx,PAD.t,h-PAD.b);
  // Fill
  ctx.beginPath(); BT.forEach((d,i)=>{{
    const x=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    i===0?ctx.moveTo(x,y0):ctx.lineTo(x,mY(d.dd_bh||0,mn,mx,PAD.t,h-PAD.b));
  }});
  ctx.fillStyle='rgba(0,144,255,.12)'; ctx.fill();
  drawLine(ctx,BT,'dd_bh','#0090ff',w,h,PAD,mn,mx,1.2);
  drawLine(ctx,BT,'dd_strat','#00ff9d',w,h,PAD,mn,mx,1.5);
}}

function drawProba(){{
  const{{ctx,w,h}}=getCtx('probaChart',260);
  const PAD={{l:50,r:16,t:12,b:30}};
  drawGrid(ctx,w,h,PAD,0,1,4,'');
  // Zones achat
  BT.forEach((d,i)=>{{
    if(!d.pos||i>=BT.length-1)return;
    const x1=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    const x2=PAD.l+((i+1)/(BT.length-1))*(w-PAD.l-PAD.r);
    ctx.fillStyle='rgba(0,255,157,.12)'; ctx.fillRect(x1,PAD.t,x2-x1,h-PAD.t-PAD.b);
  }});
  // Seuil
  const ys=mY(SBEST,0,1,PAD.t,h-PAD.b);
  ctx.strokeStyle='#00ff9d88'; ctx.lineWidth=1.5; ctx.setLineDash([5,5]);
  ctx.beginPath(); ctx.moveTo(PAD.l,ys); ctx.lineTo(w-PAD.r,ys); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#00ff9d'; ctx.font='10px IBM Plex Mono';
  ctx.fillText('Seuil '+SBEST.toFixed(2),PAD.l+4,ys-4);
  // Proba line
  drawLine(ctx,BT,'proba','#dde6f0',w,h,PAD,0,1,1.2);
  // Dates
  const step=Math.ceil(BT.length/8);
  ctx.fillStyle='#3d5a73'; ctx.font='9px IBM Plex Mono';
  BT.forEach((d,i)=>{{if(i%step===0){{
    const x=PAD.l+(i/(BT.length-1))*(w-PAD.l-PAD.r);
    ctx.fillText(d.date.slice(0,7),x-18,h-6);
  }}}});
}}

function drawHist(){{
  const{{ctx,w,h}}=getCtx('histChart',200);
  const PAD={{l:50,r:16,t:16,b:30}};
  const counts=HIST.counts; const edges=HIST.edges;
  const maxC=Math.max(...counts);
  ctx.fillStyle='#0a0f16'; ctx.fillRect(0,0,w,h);
  const bw=(w-PAD.l-PAD.r)/counts.length;
  counts.forEach((c,i)=>{{
    const x=PAD.l+i*bw;
    const h2=(c/maxC)*(h-PAD.t-PAD.b);
    const mid=(edges[i]+edges[i+1])/2;
    const col=mid>=SBEST?'#00ff9d99':'#3d5a73';
    ctx.fillStyle=col;
    ctx.fillRect(x+1,h-PAD.b-h2,bw-2,h2);
  }});
  // Axe
  ctx.fillStyle='#3d5a73'; ctx.font='9px IBM Plex Mono';
  [0,.25,.5,.75,1].forEach(v=>{{
    const x=PAD.l+v*(w-PAD.l-PAD.r);
    ctx.fillText(v.toFixed(2),x-10,h-6);
  }});
  // Seuil
  const xs=PAD.l+SBEST*(w-PAD.l-PAD.r);
  ctx.strokeStyle='#00ff9d'; ctx.lineWidth=2; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(xs,PAD.t); ctx.lineTo(xs,h-PAD.b); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#00ff9d'; ctx.fillText('Seuil',xs+4,PAD.t+12);
}}

function drawSeuils(){{
  const grid=document.getElementById('seuilGrid');
  grid.innerHTML='';
  SEUILS.forEach(s=>{{
    const active=Math.abs(s.seuil-SBEST)<0.001?'active':'';
    grid.innerHTML+=`<div class="seuil-btn ${{active}}" onclick="showSeuil(${{s.seuil}})">
      <div class="sb-val">${{s.seuil.toFixed(2)}}</div>
      <div class="sb-sub">Sharpe ${{s.sharpe.toFixed(2)}}</div>
      <div class="sb-sub">Ret ${{s.ret.toFixed(1)}}%/an</div>
    </div>`;
  }});
  showSeuil(SBEST);
}}

function showSeuil(s){{
  const d=SEUILS.find(x=>Math.abs(x.seuil-s)<0.001);
  if(!d)return;
  document.querySelectorAll('.seuil-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.seuil-btn').forEach(b=>{{
    if(b.querySelector('.sb-val').textContent===s.toFixed(2)) b.classList.add('active');
  }});
  document.getElementById('seuilDetail').innerHTML=`
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0">
      <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:9px;color:var(--muted);letter-spacing:1px">SHARPE</div>
        <div style="font-size:22px;font-weight:800;color:${{d.sharpe>1?'var(--green)':'var(--yellow)'}}">${{d.sharpe.toFixed(3)}}</div>
      </div>
      <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:9px;color:var(--muted);letter-spacing:1px">RET ANN.</div>
        <div style="font-size:22px;font-weight:800;color:${{d.ret>=0?'var(--green)':'var(--red)'}}">${{d.ret>=0?'+':''}}${{d.ret.toFixed(1)}}%</div>
      </div>
      <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:9px;color:var(--muted);letter-spacing:1px">MAX DD</div>
        <div style="font-size:22px;font-weight:800;color:var(--red)">${{d.mdd.toFixed(1)}}%</div>
      </div>
      <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:9px;color:var(--muted);letter-spacing:1px">INVESTI</div>
        <div style="font-size:22px;font-weight:800;color:var(--blue)">${{d.investi.toFixed(0)}}%</div>
      </div>
    </div>`;
}}

function drawSharpe(){{
  const{{ctx,w,h}}=getCtx('sharpeChart',200);
  const PAD={{l:50,r:16,t:16,b:30}};
  const mn=Math.min(0,...SEUILS.map(s=>s.sharpe))-0.1;
  const mx=Math.max(...SEUILS.map(s=>s.sharpe))+0.1;
  drawGrid(ctx,w,h,PAD,mn,mx,4,'');
  const bw=(w-PAD.l-PAD.r)/SEUILS.length;
  SEUILS.forEach((s,i)=>{{
    const x=PAD.l+i*bw;
    const y0=mY(0,mn,mx,PAD.t,h-PAD.b);
    const y1=mY(s.sharpe,mn,mx,PAD.t,h-PAD.b);
    const active=Math.abs(s.seuil-SBEST)<0.001;
    ctx.fillStyle=active?'#00ff9d':(s.sharpe>0?'#00ff9d66':'#ff2d5b66');
    ctx.fillRect(x+2,Math.min(y0,y1),bw-4,Math.abs(y0-y1));
    ctx.fillStyle='#3d5a73'; ctx.font='9px IBM Plex Mono';
    ctx.fillText(s.seuil.toFixed(2),x+2,h-6);
  }});
}}

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
    print("  BACKTEST LONG ONLY — S&P 500")
    print("=" * 55)

    df          = charger_donnees()
    mod, params = charger_modele_hausse()
    print(f"\nDataset : {len(df)}j ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")

    print("\nCalcul des probabilités...")
    probas = predire_historique(df, mod)

    print("\nBacktest multi-seuils...")
    seuils   = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]
    resultats = {}

    print(f"\n  {'Seuil':>6} | {'Ret Ann':>8} | {'Sharpe':>7} | {'MDD':>7} | {'Investis':>9}")
    print(f"  {'─'*50}")

    for seuil in seuils:
        bt = backtest_seuil(df, probas, seuil)
        m  = calculer_metriques(bt)
        resultats[seuil] = {'bt': bt, 'metriques': m}
        flag = " ← MEILLEUR SHARPE" if seuil == max(
            resultats, key=lambda s: resultats[s]['metriques']['sharpe_strat']
        ) else ""
        print(f"  {seuil:>6.2f} | {m['ret_ann_strat']:>+7.2f}% | "
              f"{m['sharpe_strat']:>7.3f} | {m['mdd_strat']:>6.1f}% | "
              f"{m['jours_investis']:>8.1f}%{flag}")

    # Meilleur seuil selon Sharpe
    seuil_best = max(resultats, key=lambda s: resultats[s]['metriques']['sharpe_strat'])
    bt_best    = resultats[seuil_best]['bt']
    m_best     = resultats[seuil_best]['metriques']

    print(f"\n  Seuil optimal : {seuil_best:.2f} (Sharpe={m_best['sharpe_strat']})")
    afficher_resultats(seuil_best, m_best)

    print("\nGénération du rapport HTML...")
    resultats_m = {s: resultats[s]['metriques'] for s in resultats}
    html = generer_html(resultats_m, bt_best, m_best, seuil_best, probas, df)

    chemin = "outputs/backtest_long_only.html"
    with open(chemin, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n  ✓ Rapport → {chemin}")
    os.system(f'start "" "{os.path.abspath(chemin)}"')