# ============================================================
# dashboard.py — Dashboard HTML Professionnel
# Smart Risk Manager — Multi-Actifs
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

DB_PATH = "data/market_data.db"
os.makedirs("outputs", exist_ok=True)

print("=" * 55)
print("  GENERATION DASHBOARD")
print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print("=" * 55)

# ─────────────────────────────────────────
# 1. CHARGER LES DONNEES
# ─────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

sp500   = pd.read_sql("SELECT * FROM sp500_features ORDER BY date",
                       conn, index_col='date', parse_dates=['date'])
vix     = pd.read_sql("SELECT * FROM vix_features ORDER BY date",
                       conn, index_col='date', parse_dates=['date'])
btc     = pd.read_sql("SELECT * FROM bitcoin_features ORDER BY date",
                       conn, index_col='date', parse_dates=['date'])
gold    = pd.read_sql("SELECT * FROM gold_features ORDER BY date",
                       conn, index_col='date', parse_dates=['date'])
dxy     = pd.read_sql("SELECT * FROM dxy_features ORDER BY date",
                       conn, index_col='date', parse_dates=['date'])
conn.close()

print(f"  S&P500  : {len(sp500)} jours")
print(f"  VIX     : {len(vix)} jours")
print(f"  Bitcoin : {len(btc)} jours")
print(f"  Gold    : {len(gold)} jours")
print(f"  DXY     : {len(dxy)} jours")

# ─────────────────────────────────────────
# 2. SIGNAL DU JOUR
# ─────────────────────────────────────────
def regime_vix(v):
    if v > 30:   return "DANGER",  "#ff2d5b"
    elif v > 20: return "STRESS",  "#ffd000"
    else:        return "CALME",   "#00ff9d"

last_sp  = sp500.iloc[-1]
last_vix = vix.iloc[-1]
last_btc = btc.iloc[-1]
last_gold= gold.iloc[-1]
last_dxy = dxy.iloc[-1]

vix_val        = last_vix['close']
regime, reg_col= regime_vix(vix_val)
sp_prix        = last_sp['close']
sp_ret         = last_sp['rendement'] * 100
sp_rsi         = last_sp['rsi']
sp_macd        = last_sp['macd']
btc_prix       = last_btc['close']
btc_ret        = last_btc['rendement'] * 100
gold_prix      = last_gold['close']
gold_ret       = last_gold['rendement'] * 100
dxy_prix       = last_dxy['close']
dxy_ret        = last_dxy['rendement'] * 100
date_maj       = sp500.index[-1].strftime('%d/%m/%Y')

# Signal de trading
if regime == "CALME" and sp_rsi < 70 and sp_macd > 0:
    signal, sig_col, sig_icon = "ACHAT", "#00ff9d", "▲"
elif regime == "DANGER" or sp_rsi > 75:
    signal, sig_col, sig_icon = "STOP",  "#ff2d5b", "✕"
else:
    signal, sig_col, sig_icon = "NEUTRE","#ffd000", "→"

print(f"\n  Signal du jour : {signal} | Regime : {regime}")
print(f"  S&P500 : {sp_prix:,.2f} ({sp_ret:+.2f}%)")
print(f"  VIX    : {vix_val:.2f}")

# ─────────────────────────────────────────
# 3. METRIQUES HISTORIQUES
# ─────────────────────────────────────────
def sharpe(serie):
    r = serie.dropna()
    return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0

def max_dd(prices):
    peak = prices.cummax()
    return ((prices - peak) / peak).min() * 100

def var95(serie):
    return np.percentile(serie.dropna(), 5) * 100

# Backtest simple
sp_ret_series = sp500['rendement'].dropna()
sp_sharpe     = round(sharpe(sp_ret_series), 3)
sp_maxdd      = round(max_dd(sp500['close']), 2)
sp_var95      = round(var95(sp_ret_series), 3)
sp_vol        = round(sp_ret_series.std() * np.sqrt(252) * 100, 2)
sp_perf_1an   = round((sp500['close'].iloc[-1] / sp500['close'].iloc[-252] - 1) * 100, 2) \
                if len(sp500) > 252 else 0
sp_perf_ytd   = round((sp500['close'].iloc[-1] / sp500[sp500.index.year == datetime.now().year]['close'].iloc[0] - 1) * 100, 2) \
                if len(sp500[sp500.index.year == datetime.now().year]) > 0 else 0

btc_perf_1an  = round((btc['close'].iloc[-1] / btc['close'].iloc[-252] - 1) * 100, 2) \
                if len(btc) > 252 else 0
gold_perf_1an = round((gold['close'].iloc[-1] / gold['close'].iloc[-252] - 1) * 100, 2) \
                if len(gold) > 252 else 0

# ─────────────────────────────────────────
# 4. SERIES JSON POUR GRAPHIQUES
# ─────────────────────────────────────────
N = 504  # 2 ans

def serie_json(df, col, n=N):
    sub = df[col].dropna().tail(n)
    return json.dumps([
        {"d": str(i.date()), "v": round(float(v), 4)}
        for i, v in sub.items()
    ])

def serie_json2(df, col1, col2, n=N):
    sub = df[[col1, col2]].dropna().tail(n)
    return json.dumps([
        {"d": str(i.date()),
         "v1": round(float(r[col1]), 4),
         "v2": round(float(r[col2]), 4)}
        for i, r in sub.iterrows()
    ])

sp_prix_json   = serie_json(sp500, 'close')
sp_rsi_json    = serie_json(sp500, 'rsi')
sp_macd_json   = serie_json2(sp500, 'macd', 'macd_signal')
vix_json       = serie_json(vix, 'close')
btc_json       = serie_json(btc, 'close')
gold_json      = serie_json(gold, 'close')
dxy_json       = serie_json(dxy, 'close')

# Correlations rolling 60j
dates_communes = sp500.index.intersection(btc.index)\
                            .intersection(gold.index)\
                            .intersection(dxy.index)
df_corr = pd.DataFrame({
    'sp500': sp500.loc[dates_communes, 'rendement'],
    'btc'  : btc.loc[dates_communes, 'rendement'],
    'gold' : gold.loc[dates_communes, 'rendement'],
    'dxy'  : dxy.loc[dates_communes, 'rendement'],
}).dropna().tail(N)

corr_btc  = df_corr['sp500'].rolling(60).corr(df_corr['btc']).dropna()
corr_gold = df_corr['sp500'].rolling(60).corr(df_corr['gold']).dropna()
corr_dxy  = df_corr['sp500'].rolling(60).corr(df_corr['dxy']).dropna()

corr_json = json.dumps([
    {"d": str(i.date()),
     "btc":  round(float(corr_btc.get(i, 0)), 3),
     "gold": round(float(corr_gold.get(i, 0)), 3),
     "dxy":  round(float(corr_dxy.get(i, 0)), 3)}
    for i in corr_btc.index
])

# Distribution des rendements SP500
rets = sp_ret_series.tail(504).values
hist_counts, hist_edges = np.histogram(rets, bins=40)
hist_json = json.dumps({
    "counts": hist_counts.tolist(),
    "edges":  [round(float(e), 5) for e in hist_edges.tolist()]
})

# Régimes VIX colorés
vix_tail = vix['close'].tail(N)
regimes_json = json.dumps([
    {"d": str(i.date()), "v": round(float(v), 2),
     "r": 2 if v > 30 else (1 if v > 20 else 0)}
    for i, v in vix_tail.items()
])

print("  Données JSON préparées ✓")

# ─────────────────────────────────────────
# 5. GENERATION HTML
# ─────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Smart Risk Manager — Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;700&family=Manrope:wght@300;400;600;800;900&display=swap" rel="stylesheet"/>
<style>
:root{{
  --bg:#05080c;--bg2:#0a0f16;--bg3:#111a24;--border:#162030;
  --green:#00ff9d;--red:#ff2d5b;--yellow:#ffd000;
  --blue:#0090ff;--cyan:#00d4ff;--purple:#b36eff;--orange:#ff7700;
  --text:#dde6f0;--muted:#3d5a73;--muted2:#263d52;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'Manrope',sans-serif;min-height:100vh}}
body::before{{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 70% 40% at 0% 0%,rgba(0,255,157,.05) 0,transparent 60%),
    radial-gradient(ellipse 50% 70% at 100% 100%,rgba(0,144,255,.04) 0,transparent 60%),
    radial-gradient(ellipse 40% 50% at 50% 50%,rgba(179,110,255,.02) 0,transparent 70%)}}

/* ── HEADER ── */
.hdr{{position:relative;z-index:10;padding:18px 36px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:rgba(10,15,22,.9);backdrop-filter:blur(20px)}}
.logo-wrap{{display:flex;align-items:center;gap:14px}}
.logo-icon{{width:36px;height:36px;border-radius:10px;
  background:linear-gradient(135deg,var(--green),var(--cyan));
  display:flex;align-items:center;justify-content:center;
  font-size:16px;font-weight:900;color:#05080c}}
.logo-txt{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;
  letter-spacing:3px;color:var(--text);text-transform:uppercase}}
.logo-sub{{font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:var(--muted);letter-spacing:1px;margin-top:2px}}
.hdr-r{{display:flex;align-items:center;gap:24px}}
.hdr-badge{{font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:var(--muted);letter-spacing:1px;padding:6px 12px;
  border:1px solid var(--border);border-radius:6px}}
.live-dot{{width:6px;height:6px;border-radius:50%;background:var(--green);
  animation:pulse 2s infinite;display:inline-block;margin-right:6px}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.5;transform:scale(1.3)}}}}

/* ── SIGNAL HERO ── */
.hero{{position:relative;z-index:5;padding:28px 36px;
  border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,rgba(10,15,22,.6) 0,transparent 100%)}}
.hero-grid{{display:grid;grid-template-columns:280px 1fr;gap:24px;align-items:center}}
.signal-card{{background:var(--bg2);border:1px solid {reg_col}44;
  border-radius:16px;padding:28px;position:relative;overflow:hidden}}
.signal-card::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 80% at 50% 0%,{reg_col}12 0,transparent 70%);
  pointer-events:none}}
.sig-lbl{{font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:16px}}
.sig-val{{font-family:'Manrope',sans-serif;font-size:40px;font-weight:900;
  color:{sig_col};line-height:1;margin-bottom:6px}}
.sig-icon{{font-size:20px}}
.sig-regime{{display:inline-flex;align-items:center;gap:6px;
  padding:5px 12px;border-radius:6px;
  background:{reg_col}22;border:1px solid {reg_col}44;
  font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:{reg_col};font-weight:600;letter-spacing:1px;margin-bottom:14px}}
.sig-conf{{font-size:11px;color:var(--muted)}}

.actifs-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.actif-card{{background:var(--bg2);border:1px solid var(--border);
  border-radius:12px;padding:18px;position:relative}}
.actif-nom{{font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
.actif-prix{{font-family:'Manrope',sans-serif;font-size:22px;font-weight:800;
  color:var(--text);margin-bottom:4px}}
.actif-ret{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600}}
.actif-perf{{font-size:10px;color:var(--muted);margin-top:6px}}

/* ── TABS ── */
.tabs{{display:flex;padding:0 36px;background:var(--bg2);
  border-bottom:1px solid var(--border);position:relative;z-index:10}}
.tab{{padding:15px 20px;font-family:'IBM Plex Mono',monospace;font-size:9px;
  font-weight:600;letter-spacing:2px;text-transform:uppercase;
  color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;
  transition:all .2s;user-select:none;white-space:nowrap}}
.tab:hover{{color:var(--text)}}
.tab.on{{color:var(--green);border-bottom-color:var(--green)}}
.panel{{display:none;position:relative;z-index:5;
  animation:fi .3s ease}}
.panel.on{{display:block}}
@keyframes fi{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
.page{{padding:28px 36px;max-width:1600px;margin:0 auto}}

/* ── GRILLES ── */
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:18px}}
.g4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}}

/* ── KPI ── */
.kpi{{background:var(--bg2);border:1px solid var(--border);
  border-radius:12px;padding:20px;position:relative;overflow:hidden}}
.kpi::after{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  border-radius:2px 2px 0 0}}
.kpi.g::after{{background:linear-gradient(90deg,var(--green),var(--cyan))}}
.kpi.b::after{{background:linear-gradient(90deg,var(--blue),var(--purple))}}
.kpi.y::after{{background:var(--yellow)}}
.kpi.r::after{{background:var(--red)}}
.kpi.o::after{{background:var(--orange)}}
.kpi-lbl{{font-family:'IBM Plex Mono',monospace;font-size:8px;
  color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
.kpi-val{{font-family:'Manrope',sans-serif;font-size:28px;font-weight:900;line-height:1}}
.kpi-sub{{font-size:10px;color:var(--muted);margin-top:6px}}
.kpi-sub b{{color:var(--text)}}
.green{{color:var(--green)}} .red{{color:var(--red)}}
.blue{{color:var(--blue)}}  .yellow{{color:var(--yellow)}}
.orange{{color:var(--orange)}} .cyan{{color:var(--cyan)}}

/* ── CHART BOX ── */
.cbox{{background:var(--bg2);border:1px solid var(--border);
  border-radius:12px;overflow:hidden;margin-bottom:18px}}
.cbox-hdr{{display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid var(--border)}}
.cbox-ttl{{font-family:'IBM Plex Mono',monospace;font-size:10px;
  font-weight:600;letter-spacing:1.5px;color:var(--text)}}
.leg{{display:flex;gap:14px;flex-wrap:wrap}}
.li{{display:flex;align-items:center;gap:5px;font-size:9px;color:var(--muted)}}
.ld{{width:20px;height:2px;border-radius:1px}}
canvas{{display:block;width:100%!important}}

/* ── TABLE ── */
.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl th{{font-family:'IBM Plex Mono',monospace;font-size:8px;letter-spacing:1.5px;
  text-transform:uppercase;color:var(--muted);padding:10px 14px;
  text-align:left;border-bottom:1px solid var(--border);background:var(--bg3)}}
.tbl td{{padding:10px 14px;border-bottom:1px solid var(--muted2);
  font-family:'IBM Plex Mono',monospace;font-size:11px}}
.tbl tr:last-child td{{border:none}}
.tbl tr:hover td{{background:var(--bg3)}}

/* ── GAUGE VIX ── */
.gauge-wrap{{display:flex;flex-direction:column;align-items:center;
  justify-content:center;padding:20px}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div class="logo-wrap">
    <div class="logo-icon">S</div>
    <div>
      <div class="logo-txt">Smart Risk Manager</div>
      <div class="logo-sub">Multi-Asset · ML-Powered · Live Data</div>
    </div>
  </div>
  <div class="hdr-r">
    <div class="hdr-badge"><span class="live-dot"></span>LIVE</div>
    <div class="hdr-badge">MAJ : {date_maj}</div>
    <div class="hdr-badge">S&P500 + VIX + BTC + GOLD + DXY</div>
  </div>
</div>

<!-- HERO SIGNAL -->
<div class="hero">
  <div class="hero-grid">
    <div class="signal-card">
      <div class="sig-lbl">Signal du jour</div>
      <div class="sig-val">{sig_icon} {signal}</div>
      <div class="sig-regime">● {regime}</div>
      <div class="sig-conf">
        VIX : <b style="color:{reg_col}">{vix_val:.1f}</b> &nbsp;|&nbsp;
        RSI S&P : <b>{sp_rsi:.1f}</b> &nbsp;|&nbsp;
        MACD : <b style="color:{'var(--green)' if sp_macd>0 else 'var(--red)'}">{sp_macd:.2f}</b>
      </div>
    </div>

    <div class="actifs-grid">
      <div class="actif-card">
        <div class="actif-nom">S&P 500</div>
        <div class="actif-prix">{sp_prix:,.0f}</div>
        <div class="actif-ret" style="color:{'var(--green)' if sp_ret>=0 else 'var(--red)'}">{sp_ret:+.2f}%</div>
        <div class="actif-perf">1 an : <b style="color:{'var(--green)' if sp_perf_1an>=0 else 'var(--red)'}">{sp_perf_1an:+.1f}%</b> &nbsp; YTD : <b>{sp_perf_ytd:+.1f}%</b></div>
      </div>
      <div class="actif-card">
        <div class="actif-nom">VIX — Indice de peur</div>
        <div class="actif-prix" style="color:{reg_col}">{vix_val:.2f}</div>
        <div class="actif-ret" style="color:{'var(--red)' if last_vix['rendement']*100>=0 else 'var(--green)'}">{last_vix['rendement']*100:+.2f}%</div>
        <div class="actif-perf">Régime : <b style="color:{reg_col}">{regime}</b></div>
      </div>
      <div class="actif-card">
        <div class="actif-nom">Bitcoin</div>
        <div class="actif-prix">{btc_prix:,.0f}</div>
        <div class="actif-ret" style="color:{'var(--green)' if btc_ret>=0 else 'var(--red)'}">{btc_ret:+.2f}%</div>
        <div class="actif-perf">1 an : <b style="color:{'var(--green)' if btc_perf_1an>=0 else 'var(--red)'}">{btc_perf_1an:+.1f}%</b></div>
      </div>
      <div class="actif-card">
        <div class="actif-nom">Gold &nbsp;|&nbsp; DXY</div>
        <div class="actif-prix">{gold_prix:,.0f} <span style="font-size:14px;color:var(--muted)">/ {dxy_prix:.1f}</span></div>
        <div class="actif-ret" style="color:{'var(--green)' if gold_ret>=0 else 'var(--red)'}">{gold_ret:+.2f}% <span style="color:var(--muted)">/ <span style="color:{'var(--green)' if dxy_ret>=0 else 'var(--red)'}">{dxy_ret:+.2f}%</span></span></div>
        <div class="actif-perf">1 an Gold : <b style="color:{'var(--green)' if gold_perf_1an>=0 else 'var(--red)'}">{gold_perf_1an:+.1f}%</b></div>
      </div>
    </div>
  </div>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab on" onclick="sw('marche',this)">📊 Marché</div>
  <div class="tab" onclick="sw('risque',this)">⚠️ Risque</div>
  <div class="tab" onclick="sw('technique',this)">📈 Technique</div>
  <div class="tab" onclick="sw('correlations',this)">🔗 Corrélations</div>
  <div class="tab" onclick="sw('stats',this)">📐 Statistiques</div>
</div>

<!-- ══════ TAB MARCHE ══════ -->
<div id="p-marche" class="panel on">
<div class="page">

  <div class="g4">
    <div class="kpi g">
      <div class="kpi-lbl">Sharpe S&P500</div>
      <div class="kpi-val {'green' if sp_sharpe>1 else 'yellow'}">{sp_sharpe}</div>
      <div class="kpi-sub">Rendement ajusté au risque</div>
    </div>
    <div class="kpi r">
      <div class="kpi-lbl">Drawdown Max</div>
      <div class="kpi-val red">{sp_maxdd:.1f}%</div>
      <div class="kpi-sub">Perte depuis le sommet</div>
    </div>
    <div class="kpi y">
      <div class="kpi-lbl">VaR 95%</div>
      <div class="kpi-val yellow">{sp_var95:.2f}%</div>
      <div class="kpi-sub">Perte max 95% de confiance</div>
    </div>
    <div class="kpi b">
      <div class="kpi-lbl">Volatilité Annuelle</div>
      <div class="kpi-val blue">{sp_vol:.1f}%</div>
      <div class="kpi-sub">Écart-type annualisé</div>
    </div>
  </div>

  <div class="cbox">
    <div class="cbox-hdr">
      <span class="cbox-ttl">S&P 500 — PRIX (2 ans)</span>
      <div class="leg">
        <div class="li"><div class="ld" style="background:var(--green)"></div>Cours</div>
        <div class="li"><div class="ld" style="background:var(--blue);opacity:.6"></div>MA20</div>
        <div class="li"><div class="ld" style="background:var(--orange);opacity:.6"></div>MA50</div>
      </div>
    </div>
    <canvas id="spChart" height="260"></canvas>
  </div>

  <div class="g2">
    <div class="cbox">
      <div class="cbox-hdr">
        <span class="cbox-ttl">BITCOIN — PRIX (2 ans)</span>
      </div>
      <canvas id="btcChart" height="200"></canvas>
    </div>
    <div class="cbox">
      <div class="cbox-hdr">
        <span class="cbox-ttl">GOLD & DXY — PRIX (2 ans)</span>
        <div class="leg">
          <div class="li"><div class="ld" style="background:var(--yellow)"></div>Gold</div>
          <div class="li"><div class="ld" style="background:var(--purple)"></div>DXY</div>
        </div>
      </div>
      <canvas id="goldDxyChart" height="200"></canvas>
    </div>
  </div>

  <div class="cbox">
    <div class="cbox-hdr">
      <span class="cbox-ttl">PERFORMANCE COMPARATIVE — 2 ANS (base 100)</span>
      <div class="leg">
        <div class="li"><div class="ld" style="background:var(--green)"></div>S&P500</div>
        <div class="li"><div class="ld" style="background:var(--orange)"></div>BTC</div>
        <div class="li"><div class="ld" style="background:var(--yellow)"></div>Gold</div>
        <div class="li"><div class="ld" style="background:var(--purple)"></div>DXY</div>
      </div>
    </div>
    <canvas id="perfChart" height="220"></canvas>
  </div>

</div>
</div>

<!-- ══════ TAB RISQUE ══════ -->
<div id="p-risque" class="panel">
<div class="page">

  <div class="g2">
    <div class="cbox">
      <div class="cbox-hdr">
        <span class="cbox-ttl">VIX — INDICE DE PEUR (2 ans)</span>
        <div class="leg">
          <div class="li"><div class="ld" style="background:var(--green)"></div>Calme &lt;20</div>
          <div class="li"><div class="ld" style="background:var(--yellow)"></div>Stress 20-30</div>
          <div class="li"><div class="ld" style="background:var(--red)"></div>Danger &gt;30</div>
        </div>
      </div>
      <canvas id="vixChart" height="240"></canvas>
    </div>

    <div class="cbox">
      <div class="cbox-hdr"><span class="cbox-ttl">DISTRIBUTION DES RENDEMENTS S&P500</span></div>
      <canvas id="histChart" height="240"></canvas>
    </div>
  </div>

  <div class="g3">
    <div class="kpi r">
      <div class="kpi-lbl">VaR 95% (1 jour)</div>
      <div class="kpi-val red">{sp_var95:.2f}%</div>
      <div class="kpi-sub">Si on perd, on perd au max ça dans 95% des cas</div>
    </div>
    <div class="kpi r">
      <div class="kpi-lbl">Drawdown Maximum</div>
      <div class="kpi-val red">{sp_maxdd:.1f}%</div>
      <div class="kpi-sub">Pire perte depuis un sommet (historique)</div>
    </div>
    <div class="kpi b">
      <div class="kpi-lbl">VIX actuel</div>
      <div class="kpi-val" style="color:{reg_col}">{vix_val:.2f}</div>
      <div class="kpi-sub">Régime : <b style="color:{reg_col}">{regime}</b></div>
    </div>
  </div>

  <div class="cbox">
    <div class="cbox-hdr"><span class="cbox-ttl">TABLEAU DES MÉTRIQUES DE RISQUE PAR ACTIF</span></div>
    <table class="tbl">
      <thead>
        <tr>
          <th>Actif</th>
          <th>Prix actuel</th>
          <th>Ret. 1j</th>
          <th>Perf. 1 an</th>
          <th>Volatilité ann.</th>
          <th>Sharpe</th>
          <th>VaR 95%</th>
          <th>Max DD</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="color:var(--green)">S&P 500</td>
          <td>{sp_prix:,.2f}</td>
          <td style="color:{'var(--green)' if sp_ret>=0 else 'var(--red)'}">{sp_ret:+.2f}%</td>
          <td style="color:{'var(--green)' if sp_perf_1an>=0 else 'var(--red)'}">{sp_perf_1an:+.1f}%</td>
          <td>{sp_vol:.1f}%</td>
          <td>{sp_sharpe}</td>
          <td class="red">{sp_var95:.3f}%</td>
          <td class="red">{sp_maxdd:.1f}%</td>
        </tr>
        <tr>
          <td style="color:var(--orange)">Bitcoin</td>
          <td>{btc_prix:,.0f}</td>
          <td style="color:{'var(--green)' if btc_ret>=0 else 'var(--red)'}">{btc_ret:+.2f}%</td>
          <td style="color:{'var(--green)' if btc_perf_1an>=0 else 'var(--red)'}">{btc_perf_1an:+.1f}%</td>
          <td>{round(btc['rendement'].std()*np.sqrt(252)*100,1)}%</td>
          <td>{round(sharpe(btc['rendement']),3)}</td>
          <td class="red">{round(var95(btc['rendement']),3)}%</td>
          <td class="red">{round(max_dd(btc['close']),1)}%</td>
        </tr>
        <tr>
          <td style="color:var(--yellow)">Gold</td>
          <td>{gold_prix:,.2f}</td>
          <td style="color:{'var(--green)' if gold_ret>=0 else 'var(--red)'}">{gold_ret:+.2f}%</td>
          <td style="color:{'var(--green)' if gold_perf_1an>=0 else 'var(--red)'}">{gold_perf_1an:+.1f}%</td>
          <td>{round(gold['rendement'].std()*np.sqrt(252)*100,1)}%</td>
          <td>{round(sharpe(gold['rendement']),3)}</td>
          <td class="red">{round(var95(gold['rendement']),3)}%</td>
          <td class="red">{round(max_dd(gold['close']),1)}%</td>
        </tr>
        <tr>
          <td style="color:var(--purple)">DXY</td>
          <td>{dxy_prix:.3f}</td>
          <td style="color:{'var(--green)' if dxy_ret>=0 else 'var(--red)'}">{dxy_ret:+.2f}%</td>
          <td>—</td>
          <td>{round(dxy['rendement'].std()*np.sqrt(252)*100,1)}%</td>
          <td>{round(sharpe(dxy['rendement']),3)}</td>
          <td class="red">{round(var95(dxy['rendement']),3)}%</td>
          <td class="red">{round(max_dd(dxy['close']),1)}%</td>
        </tr>
      </tbody>
    </table>
  </div>

</div>
</div>

<!-- ══════ TAB TECHNIQUE ══════ -->
<div id="p-technique" class="panel">
<div class="page">

  <div class="g4">
    <div class="kpi {'g' if sp_rsi < 70 else 'r'}">
      <div class="kpi-lbl">RSI S&P500 (14j)</div>
      <div class="kpi-val {'green' if 30<sp_rsi<70 else 'red'}">{sp_rsi:.1f}</div>
      <div class="kpi-sub">{'Suracheté !' if sp_rsi>70 else ('Survendu !' if sp_rsi<30 else 'Zone neutre')}</div>
    </div>
    <div class="kpi {'g' if sp_macd>0 else 'r'}">
      <div class="kpi-lbl">MACD</div>
      <div class="kpi-val {'green' if sp_macd>0 else 'red'}">{sp_macd:.2f}</div>
      <div class="kpi-sub">{'Signal haussier' if sp_macd>0 else 'Signal baissier'}</div>
    </div>
    <div class="kpi b">
      <div class="kpi-lbl">MA Ratio (20/50)</div>
      <div class="kpi-val {'green' if last_sp['ma_ratio']>1 else 'red'}">{last_sp['ma_ratio']:.4f}</div>
      <div class="kpi-sub">{'Golden Cross ▲' if last_sp['ma_ratio']>1 else 'Death Cross ▼'}</div>
    </div>
    <div class="kpi y">
      <div class="kpi-lbl">Vol 20j S&P500</div>
      <div class="kpi-val yellow">{round(last_sp['vol_20j']*100,2)}%</div>
      <div class="kpi-sub">Volatilité réalisée à court terme</div>
    </div>
  </div>

  <div class="cbox">
    <div class="cbox-hdr">
      <span class="cbox-ttl">RSI S&P500 — 2 ANS</span>
      <div class="leg">
        <div class="li"><div class="ld" style="background:var(--red)"></div>Suracheté &gt;70</div>
        <div class="li"><div class="ld" style="background:var(--green)"></div>Survendu &lt;30</div>
      </div>
    </div>
    <canvas id="rsiChart" height="200"></canvas>
  </div>

  <div class="cbox">
    <div class="cbox-hdr">
      <span class="cbox-ttl">MACD S&P500 — 2 ANS</span>
      <div class="leg">
        <div class="li"><div class="ld" style="background:var(--green)"></div>MACD</div>
        <div class="li"><div class="ld" style="background:var(--red)"></div>Signal</div>
      </div>
    </div>
    <canvas id="macdChart" height="180"></canvas>
  </div>

</div>
</div>

<!-- ══════ TAB CORRELATIONS ══════ -->
<div id="p-correlations" class="panel">
<div class="page">

  <div class="g3">
    <div class="kpi g">
      <div class="kpi-lbl">Corr S&P500 / BTC (60j)</div>
      <div class="kpi-val cyan">{round(float(corr_btc.iloc[-1]),3) if len(corr_btc)>0 else 'N/A'}</div>
      <div class="kpi-sub">Corrélation glissante 60 jours</div>
    </div>
    <div class="kpi y">
      <div class="kpi-lbl">Corr S&P500 / Gold (60j)</div>
      <div class="kpi-val yellow">{round(float(corr_gold.iloc[-1]),3) if len(corr_gold)>0 else 'N/A'}</div>
      <div class="kpi-sub">Corrélation glissante 60 jours</div>
    </div>
    <div class="kpi b">
      <div class="kpi-lbl">Corr S&P500 / DXY (60j)</div>
      <div class="kpi-val purple">{round(float(corr_dxy.iloc[-1]),3) if len(corr_dxy)>0 else 'N/A'}</div>
      <div class="kpi-sub">Corrélation glissante 60 jours</div>
    </div>
  </div>

  <div class="cbox">
    <div class="cbox-hdr">
      <span class="cbox-ttl">CORRÉLATIONS GLISSANTES 60j — S&P500 vs ACTIFS</span>
      <div class="leg">
        <div class="li"><div class="ld" style="background:var(--cyan)"></div>vs BTC</div>
        <div class="li"><div class="ld" style="background:var(--yellow)"></div>vs Gold</div>
        <div class="li"><div class="ld" style="background:var(--purple)"></div>vs DXY</div>
      </div>
    </div>
    <canvas id="corrChart" height="240"></canvas>
  </div>

  <div class="cbox">
    <div class="cbox-hdr"><span class="cbox-ttl">MATRICE DE CORRÉLATION — RENDEMENTS</span></div>
    <div style="padding:20px">
      <canvas id="corrMatrix" height="200" style="max-width:400px;margin:0 auto;display:block"></canvas>
    </div>
  </div>

</div>
</div>

<!-- ══════ TAB STATS ══════ -->
<div id="p-stats" class="panel">
<div class="page">

  <div class="g2">
    <div class="cbox">
      <div class="cbox-hdr"><span class="cbox-ttl">STATISTIQUES DESCRIPTIVES — S&P500</span></div>
      <table class="tbl">
        <thead><tr><th>Indicateur</th><th>Valeur</th><th>Interprétation</th></tr></thead>
        <tbody>
          <tr><td>Rendement moyen quotidien</td>
              <td style="color:var(--green)">{round(sp_ret_series.mean()*100,4)}%</td>
              <td>Drift journalier positif</td></tr>
          <tr><td>Écart-type quotidien</td>
              <td class="yellow">{round(sp_ret_series.std()*100,4)}%</td>
              <td>Dispersion des rendements</td></tr>
          <tr><td>Skewness (asymétrie)</td>
              <td style="color:{'var(--red)' if sp_ret_series.skew()<0 else 'var(--green)'}">{round(sp_ret_series.skew(),4)}</td>
              <td>{'Queues gauches (pertes extrêmes)' if sp_ret_series.skew()<0 else 'Queues droites'}</td></tr>
          <tr><td>Kurtosis (aplatissement)</td>
              <td class="yellow">{round(sp_ret_series.kurtosis(),4)}</td>
              <td>{'Fat tails — queues épaisses' if sp_ret_series.kurtosis()>3 else 'Distribution normale'}</td></tr>
          <tr><td>Min (pire journée)</td>
              <td class="red">{round(sp_ret_series.min()*100,3)}%</td>
              <td>Pire chute en un jour</td></tr>
          <tr><td>Max (meilleure journée)</td>
              <td class="green">{round(sp_ret_series.max()*100,3)}%</td>
              <td>Meilleur gain en un jour</td></tr>
          <tr><td>Médiane quotidienne</td>
              <td class="green">{round(sp_ret_series.median()*100,4)}%</td>
              <td>50% des jours au-dessus</td></tr>
          <tr><td>Nb jours hausse</td>
              <td class="green">{(sp_ret_series>0).sum()} ({round((sp_ret_series>0).mean()*100,1)}%)</td>
              <td>Win rate historique</td></tr>
        </tbody>
      </table>
    </div>

    <div class="cbox">
      <div class="cbox-hdr"><span class="cbox-ttl">RÉSUMÉ RISQUE MULTI-ACTIFS</span></div>
      <table class="tbl">
        <thead><tr><th>Actif</th><th>Sharpe</th><th>Sortino</th><th>Skew</th><th>VaR95</th></tr></thead>
        <tbody>"""

for nom, df_a in [('S&P500', sp500), ('Bitcoin', btc),
                   ('Gold', gold), ('DXY', dxy)]:
    r   = df_a['rendement'].dropna()
    sh  = round(sharpe(r), 3)
    neg = r[r < 0]
    sor = round((r.mean() / neg.std()) * np.sqrt(252), 3) if len(neg) > 0 else 0
    skw = round(r.skew(), 3)
    v95 = round(var95(r), 3)
    html += f"""
          <tr>
            <td style="color:var(--green)">{nom}</td>
            <td style="color:{'var(--green)' if sh>1 else 'var(--yellow)'}">{sh}</td>
            <td style="color:{'var(--green)' if sor>1 else 'var(--yellow)'}">{sor}</td>
            <td style="color:{'var(--red)' if skw<0 else 'var(--green)'}">{skw}</td>
            <td class="red">{v95}%</td>
          </tr>"""

html += f"""
        </tbody>
      </table>
    </div>
  </div>

  <div class="cbox">
    <div class="cbox-hdr"><span class="cbox-ttl">RÉGIMES DE MARCHÉ — DISTRIBUTION HISTORIQUE</span></div>
    <div style="padding:24px;display:grid;grid-template-columns:repeat(3,1fr);gap:20px;text-align:center">
      <div>
        <div style="font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace;letter-spacing:1px;margin-bottom:10px">CALME (VIX &lt; 20)</div>
        <div style="font-size:36px;font-weight:900;color:var(--green)">{round((vix['close']<20).mean()*100,1)}%</div>
        <div style="font-size:11px;color:var(--muted);margin-top:6px">du temps</div>
      </div>
      <div>
        <div style="font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace;letter-spacing:1px;margin-bottom:10px">STRESS (20-30)</div>
        <div style="font-size:36px;font-weight:900;color:var(--yellow)">{round(((vix['close']>=20)&(vix['close']<30)).mean()*100,1)}%</div>
        <div style="font-size:11px;color:var(--muted);margin-top:6px">du temps</div>
      </div>
      <div>
        <div style="font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace;letter-spacing:1px;margin-bottom:10px">DANGER (VIX &gt; 30)</div>
        <div style="font-size:36px;font-weight:900;color:var(--red)">{round((vix['close']>=30).mean()*100,1)}%</div>
        <div style="font-size:11px;color:var(--muted);margin-top:6px">du temps</div>
      </div>
    </div>
  </div>

</div>
</div>

<script>
const SP   = {sp_prix_json};
const RSI  = {sp_rsi_json};
const MACD = {sp_macd_json};
const VIX  = {vix_json};
const BTC  = {btc_json};
const GOLD = {gold_json};
const DXY  = {dxy_json};
const CORR = {corr_json};
const HIST = {hist_json};
const REG  = {regimes_json};

function sw(n,el){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  document.getElementById('p-'+n).classList.add('on'); el.classList.add('on');
  setTimeout(()=>{{drawAll(n);}},50);
}}

const CS={{}};
function gc(id,h){{
  const el=document.getElementById(id);
  if(!el)return null;
  el.width=el.parentElement.clientWidth||800;
  el.height=h||240;
  const ctx=el.getContext('2d');
  ctx.clearRect(0,0,el.width,el.height);
  CS[id]={{ctx,w:el.width,h:el.height}};
  return CS[id];
}}

const C={{bg:'#0a0f16',border:'#162030',muted:'#3d5a73',text:'#dde6f0',
  green:'#00ff9d',red:'#ff2d5b',yellow:'#ffd000',blue:'#0090ff',
  cyan:'#00d4ff',purple:'#b36eff',orange:'#ff7700'}};

function mY(v,mn,mx,t,b){{
  if(mx===mn)return(t+b)/2;
  return t+(1-(v-mn)/(mx-mn))*(b-t);
}}

function bg(ctx,w,h){{ctx.fillStyle=C.bg;ctx.fillRect(0,0,w,h);}}

function grid(ctx,w,h,PAD,mn,mx,steps,fmt){{
  for(let i=0;i<=steps;i++){{
    const v=mn+(mx-mn)*i/steps;
    const y=mY(v,mn,mx,PAD.t,h-PAD.b);
    ctx.strokeStyle=C.border; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(w-PAD.r,y); ctx.stroke();
    ctx.fillStyle=C.muted; ctx.font='9px IBM Plex Mono'; ctx.textAlign='right';
    ctx.fillText(fmt(v),PAD.l-4,y+3);
  }}
  ctx.textAlign='left';
}}

function line(ctx,data,key,col,PAD,w,h,mn,mx,lw){{
  ctx.beginPath(); let f=true;
  data.forEach((d,i)=>{{
    const x=PAD.l+(i/(data.length-1||1))*(w-PAD.l-PAD.r);
    const y=mY(d[key],mn,mx,PAD.t,h-PAD.b);
    f?ctx.moveTo(x,y):ctx.lineTo(x,y); f=false;
  }});
  ctx.strokeStyle=col; ctx.lineWidth=lw||1.5; ctx.stroke();
}}

function fill(ctx,data,key,col,PAD,w,h,mn,mx){{
  const y0=mY(0,mn,mx,PAD.t,h-PAD.b);
  ctx.beginPath();
  data.forEach((d,i)=>{{
    const x=PAD.l+(i/(data.length-1||1))*(w-PAD.l-PAD.r);
    const y=mY(d[key],mn,mx,PAD.t,h-PAD.b);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});
  ctx.lineTo(PAD.l+(data.length-1)/(data.length-1)*(w-PAD.l-PAD.r),y0);
  ctx.lineTo(PAD.l,y0); ctx.closePath();
  ctx.fillStyle=col+'22'; ctx.fill();
}}

function xLabels(ctx,data,w,h,PAD){{
  ctx.fillStyle=C.muted; ctx.font='9px IBM Plex Mono'; ctx.textAlign='center';
  const step=Math.ceil(data.length/8);
  data.forEach((d,i)=>{{
    if(i%step===0){{
      const x=PAD.l+(i/(data.length-1||1))*(w-PAD.l-PAD.r);
      ctx.fillText(d.d.slice(0,7),x,h-4);
    }}
  }});
  ctx.textAlign='left';
}}

function drawSP(){{
  const c=gc('spChart',260); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:60,r:16,t:16,b:26}};
  const mn=Math.min(...SP.map(d=>d.v))*0.98;
  const mx=Math.max(...SP.map(d=>d.v))*1.02;
  grid(ctx,w,h,PAD,mn,mx,5,v=>v.toFixed(0));
  fill(ctx,SP,'v',C.green,PAD,w,h,mn,mx);
  line(ctx,SP,'v',C.green,PAD,w,h,mn,mx,2);
  xLabels(ctx,SP,w,h,PAD);
}}

function drawBTC(){{
  const c=gc('btcChart',200); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:70,r:16,t:12,b:26}};
  const mn=Math.min(...BTC.map(d=>d.v))*0.97;
  const mx=Math.max(...BTC.map(d=>d.v))*1.03;
  grid(ctx,w,h,PAD,mn,mx,4,v=>v>=1000?Math.round(v/1000)+'k':v.toFixed(0));
  fill(ctx,BTC,'v',C.orange,PAD,w,h,mn,mx);
  line(ctx,BTC,'v',C.orange,PAD,w,h,mn,mx,1.8);
  xLabels(ctx,BTC,w,h,PAD);
}}

function drawGoldDxy(){{
  const c=gc('goldDxyChart',200); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:60,r:60,t:12,b:26}};
  // Gold left
  const mn1=Math.min(...GOLD.map(d=>d.v))*0.97;
  const mx1=Math.max(...GOLD.map(d=>d.v))*1.03;
  grid(ctx,w,h,PAD,mn1,mx1,4,v=>v.toFixed(0));
  line(ctx,GOLD,'v',C.yellow,PAD,w,h,mn1,mx1,1.8);
  // DXY right axis
  const mn2=Math.min(...DXY.map(d=>d.v))*0.99;
  const mx2=Math.max(...DXY.map(d=>d.v))*1.01;
  ctx.fillStyle=C.muted; ctx.font='9px IBM Plex Mono'; ctx.textAlign='left';
  [0,.25,.5,.75,1].forEach(t=>{{
    const v=mn2+(mx2-mn2)*t;
    const y=mY(v,mn2,mx2,PAD.t,h-PAD.b);
    ctx.fillText(v.toFixed(1),w-PAD.r+4,y+3);
  }});
  ctx.textAlign='left';
  line(ctx,DXY,'v',C.purple,{{l:PAD.l,r:PAD.r,t:PAD.t,b:PAD.b}},w,h,mn2,mx2,1.5);
  xLabels(ctx,GOLD,w,h,PAD);
}}

function drawPerf(){{
  const c=gc('perfChart',220); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:50,r:16,t:12,b:26}};
  const base=(d,arr)=>arr.map(x=>{{return{{d:x.d,v:x.v/arr[0].v*100}}}});
  const sp2=base(null,SP); const bt2=base(null,BTC);
  const go2=base(null,GOLD); const dx2=base(null,DXY);
  const all=[...sp2,...bt2,...go2,...dx2];
  const mn=Math.min(...all.map(d=>d.v))*0.97;
  const mx=Math.max(...all.map(d=>d.v))*1.03;
  grid(ctx,w,h,PAD,mn,mx,5,v=>v.toFixed(0));
  // line 100
  const y100=mY(100,mn,mx,PAD.t,h-PAD.b);
  ctx.strokeStyle='#263d52'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(PAD.l,y100); ctx.lineTo(w-PAD.r,y100); ctx.stroke();
  ctx.setLineDash([]);
  line(ctx,sp2,'v',C.green,PAD,w,h,mn,mx,2);
  line(ctx,bt2,'v',C.orange,PAD,w,h,mn,mx,1.5);
  line(ctx,go2,'v',C.yellow,PAD,w,h,mn,mx,1.5);
  line(ctx,dx2,'v',C.purple,PAD,w,h,mn,mx,1.5);
  xLabels(ctx,SP,w,h,PAD);
}}

function drawVIX(){{
  const c=gc('vixChart',240); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:40,r:16,t:12,b:26}};
  const mn=Math.min(...VIX.map(d=>d.v))*0.9;
  const mx=Math.max(...VIX.map(d=>d.v))*1.1;
  // Zones colorées
  REG.forEach((d,i)=>{{
    if(i>=REG.length-1)return;
    const x1=PAD.l+(i/(REG.length-1))*(w-PAD.l-PAD.r);
    const x2=PAD.l+((i+1)/(REG.length-1))*(w-PAD.l-PAD.r);
    ctx.fillStyle=d.r===2?C.red+'22':(d.r===1?C.yellow+'15':C.green+'10');
    ctx.fillRect(x1,PAD.t,x2-x1,h-PAD.t-PAD.b);
  }});
  grid(ctx,w,h,PAD,mn,mx,5,v=>v.toFixed(0));
  // Lignes seuil
  [20,30].forEach((s,i)=>{{
    const y=mY(s,mn,mx,PAD.t,h-PAD.b);
    ctx.strokeStyle=i===0?C.yellow:C.red; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(w-PAD.r,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=i===0?C.yellow:C.red; ctx.font='9px IBM Plex Mono';
    ctx.fillText(s,PAD.l+4,y-3);
  }});
  line(ctx,VIX,'v',C.cyan,PAD,w,h,mn,mx,1.5);
  xLabels(ctx,VIX,w,h,PAD);
}}

function drawHist(){{
  const c=gc('histChart',240); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:40,r:16,t:16,b:30}};
  const counts=HIST.counts; const edges=HIST.edges;
  const maxC=Math.max(...counts);
  const bw=(w-PAD.l-PAD.r)/counts.length;
  counts.forEach((cnt,i)=>{{
    const mid=(edges[i]+edges[i+1])/2;
    const bh=(cnt/maxC)*(h-PAD.t-PAD.b);
    ctx.fillStyle=mid<0?C.red+'cc':C.green+'cc';
    ctx.fillRect(PAD.l+i*bw+1,h-PAD.b-bh,bw-2,bh);
  }});
  // Axe X
  ctx.fillStyle=C.muted; ctx.font='9px IBM Plex Mono'; ctx.textAlign='center';
  [-0.04,-0.02,0,0.02,0.04].forEach(v=>{{
    const x=PAD.l+(v-edges[0])/(edges[edges.length-1]-edges[0])*(w-PAD.l-PAD.r);
    ctx.fillText((v*100).toFixed(1)+'%',x,h-4);
  }});
  // Ligne 0
  const x0=PAD.l+(0-edges[0])/(edges[edges.length-1]-edges[0])*(w-PAD.l-PAD.r);
  ctx.strokeStyle=C.muted; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(x0,PAD.t); ctx.lineTo(x0,h-PAD.b); ctx.stroke();
}}

function drawRSI(){{
  const c=gc('rsiChart',200); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:35,r:16,t:12,b:26}};
  grid(ctx,w,h,PAD,0,100,4,v=>v.toFixed(0));
  // Zones
  const y30=mY(30,0,100,PAD.t,h-PAD.b);
  const y70=mY(70,0,100,PAD.t,h-PAD.b);
  ctx.fillStyle=C.red+'10'; ctx.fillRect(PAD.l,PAD.t,w-PAD.l-PAD.r,y70-PAD.t);
  ctx.fillStyle=C.green+'10'; ctx.fillRect(PAD.l,y30,w-PAD.l-PAD.r,h-PAD.b-y30);
  [30,70].forEach(v=>{{
    const y=mY(v,0,100,PAD.t,h-PAD.b);
    ctx.strokeStyle=v===70?C.red+'66':C.green+'66'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(w-PAD.r,y); ctx.stroke();
    ctx.setLineDash([]);
  }});
  line(ctx,RSI,'v',C.cyan,PAD,w,h,0,100,1.5);
  xLabels(ctx,RSI,w,h,PAD);
}}

function drawMACD(){{
  const c=gc('macdChart',180); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:45,r:16,t:12,b:26}};
  const mn=Math.min(...MACD.map(d=>Math.min(d.v1,d.v2)))*1.2;
  const mx=Math.max(...MACD.map(d=>Math.max(d.v1,d.v2)))*1.2;
  grid(ctx,w,h,PAD,mn,mx,4,v=>v.toFixed(1));
  const y0=mY(0,mn,mx,PAD.t,h-PAD.b);
  ctx.strokeStyle=C.muted; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(PAD.l,y0); ctx.lineTo(w-PAD.r,y0); ctx.stroke();
  line(ctx,MACD,'v1',C.green,PAD,w,h,mn,mx,1.8);
  line(ctx,MACD,'v2',C.red,PAD,w,h,mn,mx,1.2);
  xLabels(ctx,MACD,w,h,PAD);
}}

function drawCorr(){{
  const c=gc('corrChart',240); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const PAD={{l:40,r:16,t:12,b:26}};
  grid(ctx,w,h,PAD,-1,1,4,v=>v.toFixed(1));
  const y0=mY(0,-1,1,PAD.t,h-PAD.b);
  ctx.strokeStyle=C.border; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(PAD.l,y0); ctx.lineTo(w-PAD.r,y0); ctx.stroke();
  line(ctx,CORR,'btc',C.cyan,PAD,w,h,-1,1,1.5);
  line(ctx,CORR,'gold',C.yellow,PAD,w,h,-1,1,1.5);
  line(ctx,CORR,'dxy',C.purple,PAD,w,h,-1,1,1.5);
  xLabels(ctx,CORR,w,h,PAD);
}}

function drawCorrMatrix(){{
  const c=gc('corrMatrix',200); if(!c)return;
  const{{ctx,w,h}}=c; bg(ctx,w,h);
  const labels=['SP500','BTC','Gold','DXY'];
  const n=4; const cell=Math.min(w,h-20)/n;
  const ox=(w-n*cell)/2; const oy=20;
  ctx.font='10px IBM Plex Mono'; ctx.textAlign='center';
  labels.forEach((l,i)=>{{
    ctx.fillStyle=C.muted;
    ctx.fillText(l,ox+i*cell+cell/2,14);
    ctx.fillText(l,ox-6,oy+i*cell+cell/2+4);
  }});
  const lastCorrs=[
    [1, CORR.length>0?CORR[CORR.length-1].btc:0,
        CORR.length>0?CORR[CORR.length-1].gold:0,
        CORR.length>0?CORR[CORR.length-1].dxy:0],
    [CORR.length>0?CORR[CORR.length-1].btc:0,1,0,0],
    [CORR.length>0?CORR[CORR.length-1].gold:0,0,1,0],
    [CORR.length>0?CORR[CORR.length-1].dxy:0,0,0,1],
  ];
  lastCorrs.forEach((row,i)=>{{
    row.forEach((v,j)=>{{
      const x=ox+j*cell; const y=oy+i*cell;
      const abs=Math.abs(v);
      ctx.fillStyle=v>0?`rgba(0,255,157,${{abs*0.7}})`:
                   v<0?`rgba(255,45,91,${{abs*0.7}})`:'rgba(22,32,48,0.5)';
      ctx.fillRect(x+1,y+1,cell-2,cell-2);
      ctx.fillStyle=abs>0.4?'#fff':C.muted;
      ctx.font='9px IBM Plex Mono';
      ctx.fillText(v.toFixed(2),x+cell/2,y+cell/2+3);
    }});
  }});
}}

function drawAll(tab){{
  if(tab==='marche'){{drawSP();drawBTC();drawGoldDxy();drawPerf();}}
  if(tab==='risque'){{drawVIX();drawHist();}}
  if(tab==='technique'){{drawRSI();drawMACD();}}
  if(tab==='correlations'){{drawCorr();drawCorrMatrix();}}
}}

window.addEventListener('load',()=>{{drawSP();drawBTC();drawGoldDxy();drawPerf();}});
window.addEventListener('resize',()=>{{
  const on=document.querySelector('.tab.on');
  if(on){{
    const tab=on.onclick.toString().match(/'(\\w+)'/)[1];
    drawAll(tab);
  }}
}});
</script>
</body>
</html>"""

# ─────────────────────────────────────────
# 6. SAUVEGARDER ET OUVRIR
# ─────────────────────────────────────────
chemin = "outputs/dashboard.html"
with open(chemin, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n  Dashboard généré → {chemin}")
os.system(f'start "" "{os.path.abspath(chemin)}"')
print("\nDashboard ouvert dans le navigateur !")