# ============================================================
#  dashboard_live.py — Dashboard Live S&P500 ML Predictor
#
#  Lance : python dashboard_live.py
#  Ouvre : http://localhost:5000
#
#  Dépendances : pip install flask
# ============================================================

import sqlite3
import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
DB_PATH = "data/market_data.db"


# ─────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────

def get_conn():
    return sqlite3.connect(DB_PATH)


def safe_float(val, decimals=2):
    try:
        return round(float(val), decimals)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# DONNÉES MARCHÉ EN TEMPS RÉEL (depuis SQLite)
# ─────────────────────────────────────────────────────────────

def get_market_data():
    conn = get_conn()
    result = {}

    actifs = {
        'sp500'  : ('^GSPC',  'S&P 500'),
        'vix'    : ('^VIX',   'VIX'),
        'bitcoin': ('BTC-USD','Bitcoin'),
        'gold'   : ('GC=F',   'Or'),
        'dxy'    : ('DX-Y.NYB','Dollar DXY'),
    }

    for nom, (ticker, label) in actifs.items():
        try:
            df = pd.read_sql(
                f"SELECT date, close FROM {nom}_prices ORDER BY date DESC LIMIT 2",
                conn
            )
            if len(df) >= 2:
                prix_actuel  = safe_float(df['close'].iloc[0])
                prix_veille  = safe_float(df['close'].iloc[1])
                variation    = safe_float((prix_actuel - prix_veille) / prix_veille * 100)
                date_str     = str(df['date'].iloc[0])[:10]
            elif len(df) == 1:
                prix_actuel  = safe_float(df['close'].iloc[0])
                variation    = 0.0
                date_str     = str(df['date'].iloc[0])[:10]
            else:
                prix_actuel  = 0.0
                variation    = 0.0
                date_str     = "N/A"

            result[nom] = {
                'ticker'   : ticker,
                'label'    : label,
                'prix'     : prix_actuel,
                'variation': variation,
                'date'     : date_str,
            }
        except Exception as e:
            result[nom] = {'ticker': ticker, 'label': label,
                           'prix': 0.0, 'variation': 0.0, 'date': 'N/A'}

    conn.close()
    return result


# ─────────────────────────────────────────────────────────────
# SIGNAL ML DEPUIS LES MODÈLES SAUVEGARDÉS
# ─────────────────────────────────────────────────────────────

def get_ml_signal():
    try:
        conn = get_conn()
        df   = pd.read_sql(
            "SELECT * FROM sp500_ml_features ORDER BY date DESC LIMIT 1",
            conn, parse_dates=['date']
        )
        conn.close()

        if df.empty:
            return default_signal()

        # Enrichissement
        if 'vix_close' in df.columns:
            conn2 = get_conn()
            vix_hist = pd.read_sql(
                "SELECT date, close FROM vix_prices ORDER BY date",
                conn2, parse_dates=['date']
            ).set_index('date')
            conn2.close()

            vix_val = safe_float(df['vix_close'].iloc[0])
            vix_pct = safe_float(
                (vix_hist['close'] <= vix_val).mean() * 100
            )
        else:
            vix_val = 20.0
            vix_pct = 50.0

        regime = ('crash'  if vix_val >= 30 else
                  'stress' if vix_val >= 20 else 'calme')

        # Charger modèle
        modeles  = joblib.load(f"models/ensemble_{regime}.pkl")
        scaler   = joblib.load(f"models/scaler_{regime}.pkl")
        features = joblib.load(f"models/features_{regime}.pkl")

        feats_dispo = [f for f in features if f in df.columns]
        X           = df[feats_dispo].values.astype(np.float64)
        X_scaled    = scaler.transform(X)

        poids  = {'rf': 0.30, 'xgb': 0.50, 'lr': 0.20}
        proba  = 0.0
        for nom, m in modeles.items():
            p = m['model'].predict_proba(X_scaled)[0, 1]
            proba += p * poids.get(nom, 0.33)

        # Exposition
        if regime == 'calme':
            if   proba >= 0.65: expo = 1.0;  signal = "FORT ACHAT";  couleur = "green"
            elif proba >= 0.60: expo = 0.8;  signal = "ACHAT";       couleur = "green"
            elif proba >= 0.55: expo = 0.6;  signal = "ACHAT";       couleur = "green"
            elif proba >= 0.50: expo = 0.4;  signal = "RÉDUIT";      couleur = "amber"
            elif proba >= 0.44: expo = 0.3;  signal = "RÉDUIT";      couleur = "amber"
            else:               expo = 0.0;  signal = "NEUTRE";      couleur = "gray"
        elif regime == 'stress':
            if   proba >= 0.58: expo = 0.8;  signal = "ACHAT";       couleur = "green"
            elif proba >= 0.52: expo = 0.5;  signal = "ACHAT";       couleur = "green"
            elif proba >= 0.45: expo = 0.3;  signal = "RÉDUIT";      couleur = "amber"
            elif proba >= 0.38: expo = 0.0;  signal = "NEUTRE";      couleur = "gray"
            else:               expo = -0.2; signal = "SHORT";       couleur = "red"
        else:  # crash
            if   proba >= 0.65: expo = 0.5;  signal = "ACHAT TIMIDE";couleur = "amber"
            elif proba >= 0.55: expo = 0.2;  signal = "RÉDUIT";      couleur = "amber"
            else:               expo = 0.0;  signal = "NEUTRE";      couleur = "gray"

        # Score conviction
        score = (proba - 0.5) * 200
        if   abs(score) >= 35: conviction = "FORTE"
        elif abs(score) >= 15: conviction = "MODÉRÉE"
        else:                  conviction = "FAIBLE"

        return {
            'date'       : str(df['date'].iloc[0].date()) if hasattr(df['date'].iloc[0], 'date') else str(df['date'].iloc[0])[:10],
            'vix'        : vix_val,
            'vix_pct'    : vix_pct,
            'regime'     : regime,
            'proba'      : safe_float(proba * 100, 1),
            'exposition' : safe_float(expo * 100),
            'signal'     : signal,
            'couleur'    : couleur,
            'score'      : safe_float(score, 1),
            'conviction' : conviction,
            'modele_ok'  : True,
        }

    except FileNotFoundError:
        return default_signal()
    except Exception as e:
        print(f"  Erreur signal ML : {e}")
        return default_signal()


def default_signal():
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'vix': 0.0, 'vix_pct': 50.0, 'regime': 'calme',
        'proba': 50.0, 'exposition': 60.0,
        'signal': 'N/A', 'couleur': 'gray',
        'score': 0.0, 'conviction': 'FAIBLE', 'modele_ok': False,
    }


# ─────────────────────────────────────────────────────────────
# TENDANCE TECHNIQUE
# ─────────────────────────────────────────────────────────────

def get_tendance():
    try:
        conn = get_conn()
        sp   = pd.read_sql(
            "SELECT date, close, high, low, open, volume FROM sp500_prices ORDER BY date DESC LIMIT 300",
            conn
        ).iloc[::-1].reset_index(drop=True)
        conn.close()

        close = sp['close']
        ma20  = safe_float(close.rolling(20).mean().iloc[-1])
        ma50  = safe_float(close.rolling(50).mean().iloc[-1])
        ma200 = safe_float(close.rolling(200).mean().iloc[-1])
        prix  = safe_float(close.iloc[-1])

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = safe_float(100 - (100 / (1 + gain / loss)).iloc[-1])

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        macd_sig = macd.ewm(span=9).mean()
        macd_hist = safe_float((macd - macd_sig).iloc[-1])

        bb_std = close.rolling(20).std()
        bb_z   = safe_float((close - close.rolling(20).mean()).iloc[-1] /
                             bb_std.iloc[-1])

        mom5  = safe_float((close.iloc[-1] / close.iloc[-6] - 1) * 100, 2)

        # Scores
        def score_rsi(r):
            if r > 60: return 20
            elif r > 50: return 10
            elif r < 40: return -20
            else: return -10

        def score_prix_ma(p, ma):
            ecart = (p - ma) / ma * 100
            if ecart > 3: return 20
            elif ecart > 0: return 10
            elif ecart < -3: return -20
            else: return -10

        s_ct = score_rsi(rsi) + (20 if macd_hist > 0 else -20) + (10 if mom5 > 0 else -10)
        s_mt = score_prix_ma(prix, ma20) + score_prix_ma(prix, ma50) + (20 if ma20 > ma50 else -20)
        s_lt = score_prix_ma(prix, ma200) + (15 if ma50 > ma200 else -15) + (20 if prix > ma200 else -20)

        max_s = 60
        s_ct_n = round(s_ct / max_s * 100, 1)
        s_mt_n = round(s_mt / max_s * 100, 1)
        s_lt_n = round(s_lt / max_s * 100, 1)

        resistance = safe_float(close.rolling(20).max().iloc[-1] * 1.02)
        support_1  = safe_float(ma50 * 0.97)
        support_2  = safe_float(ma200 * 0.98)

        return {
            'ma20'       : ma20,
            'ma50'       : ma50,
            'ma200'      : ma200,
            'rsi'        : rsi,
            'macd_hist'  : macd_hist,
            'bb_zscore'  : bb_z,
            'mom5'       : mom5,
            'score_ct'   : s_ct_n,
            'score_mt'   : s_mt_n,
            'score_lt'   : s_lt_n,
            'ecart_ma20' : safe_float((prix / ma20 - 1) * 100, 2),
            'ecart_ma50' : safe_float((prix / ma50 - 1) * 100, 2),
            'ecart_ma200': safe_float((prix / ma200 - 1) * 100, 2),
            'resistance' : resistance,
            'support_1'  : support_1,
            'support_2'  : support_2,
        }
    except Exception as e:
        print(f"  Erreur tendance : {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# HISTORIQUE DES PERFORMANCES (depuis les CSV résultats)
# ─────────────────────────────────────────────────────────────

def get_perf_history():
    result = {'calme': [], 'stress': []}
    for regime in ['calme', 'stress']:
        path = f"results/comparaison_{regime}.csv"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            # Cumuls sous forme de liste pour le graphe
            for col in ['cumul_strat', 'cumul_bh', 'cumul_bin']:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill')

            dates   = df['date'].tolist() if 'date' in df.columns else list(range(len(df)))
            # Prendre 1 point tous les 5 jours pour alléger
            step    = max(1, len(df) // 50)
            indices = list(range(0, len(df), step))

            result[regime] = {
                'dates'      : [str(dates[i])[:10] for i in indices],
                'ml'         : [safe_float(df['cumul_strat'].iloc[i] * 100 - 100) for i in indices],
                'bh'         : [safe_float(df['cumul_bh'].iloc[i]    * 100 - 100) for i in indices],
                'bin'        : [safe_float(df['cumul_bin'].iloc[i]    * 100 - 100) for i in indices] if 'cumul_bin' in df.columns else [],
            }
        except Exception as e:
            print(f"  Erreur perf {regime} : {e}")
    return result


# ─────────────────────────────────────────────────────────────
# HISTORIQUE DES TENDANCES
# ─────────────────────────────────────────────────────────────

def get_tendance_history():
    path = "results/historique_tendances.csv"
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path).tail(30)
        return df.to_dict(orient='records')
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route('/api/data')
def api_data():
    marche  = get_market_data()
    signal  = get_ml_signal()
    tendance = get_tendance()
    perf    = get_perf_history()
    hist    = get_tendance_history()

    return jsonify({
        'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'marche'   : marche,
        'signal'   : signal,
        'tendance' : tendance,
        'perf'     : perf,
        'historique': hist,
    })


@app.route('/api/signal')
def api_signal():
    return jsonify(get_ml_signal())


# ─────────────────────────────────────────────────────────────
# PAGE HTML DU DASHBOARD
# ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S&P 500 ML Predictor — Live</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080c14;--bg2:#0d1320;--bg3:#111b2e;--bg4:#162035;
  --bd:#1e2d45;--bd2:#253650;
  --t:#e2e8f0;--t2:#94a3b8;--t3:#4a5568;
  --green:#22c55e;--green2:#3b6d11;
  --red:#ef4444;--red2:#a32d2d;
  --blue:#3b82f6;--blue2:#185fa5;
  --amber:#f97316;--amber2:#854f0b;
  --yellow:#eab308;
  --mono:'IBM Plex Mono',monospace;
}
html,body{background:var(--bg);color:var(--t);font-family:var(--mono);font-size:13px;min-height:100vh}
.wrap{max-width:1440px;margin:0 auto;padding:0 24px 48px}
/* TOPBAR */
.topbar{display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:0.5px solid var(--bd);margin-bottom:18px}
.logo-title{font-size:15px;font-weight:500;letter-spacing:.03em}
.logo-sub{font-size:10px;color:var(--t3);letter-spacing:.08em;text-transform:uppercase;margin-top:2px}
.live-row{display:flex;align-items:center;gap:10px}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.badge-pill{font-size:10px;background:var(--bg3);border:0.5px solid var(--bd);padding:4px 12px;border-radius:5px;color:var(--t2)}
/* GRIDS */
.g4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin-bottom:14px}
.g3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin-bottom:14px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-bottom:14px}
.g21{display:grid;grid-template-columns:2fr 1fr;gap:11px;margin-bottom:14px}
.g32{display:grid;grid-template-columns:3fr 2fr;gap:11px;margin-bottom:14px}
/* METRIC */
.mc{background:var(--bg3);border-radius:8px;padding:13px 15px}
.ml{font-size:9px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px}
.mv{font-size:24px;font-weight:500;line-height:1}
.ms{font-size:10px;color:var(--t2);margin-top:4px;display:flex;align-items:center;gap:5px}
/* CARD */
.card{background:var(--bg2);border:0.5px solid var(--bd);border-radius:10px;padding:15px 18px}
.ct{font-size:9px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px}
/* COLORS */
.green{color:var(--green)}.red{color:var(--red)}.blue{color:var(--blue)}.amber{color:var(--amber)}.yellow{color:var(--yellow)}
/* SIGNAL */
.sig-hero{text-align:center;padding:18px 8px}
.sig-tag{display:inline-flex;align-items:center;gap:7px;font-size:18px;font-weight:500;padding:9px 22px;border-radius:7px;margin-bottom:8px;border:1px solid}
.sig-green{background:rgba(34,197,94,.1);color:var(--green);border-color:rgba(34,197,94,.25)}
.sig-red  {background:rgba(239,68,68,.1);color:var(--red)  ;border-color:rgba(239,68,68,.25)}
.sig-amber{background:rgba(249,115,22,.1);color:var(--amber);border-color:rgba(249,115,22,.25)}
.sig-gray {background:rgba(148,163,184,.08);color:var(--t2);border-color:var(--bd2)}
/* PBAR */
.pbar{width:100%;height:5px;background:var(--bg4);border-radius:3px;overflow:hidden;margin:8px 0 4px}
.pfill{height:100%;border-radius:3px;transition:width 1s ease}
/* ROWS */
.row{display:flex;align-items:center;gap:9px;margin-bottom:8px}
.rl{font-size:10px;color:var(--t2);min-width:90px}
.rb{flex:1;height:5px;background:var(--bg4);border-radius:3px;overflow:hidden}
.rf{height:100%;border-radius:3px}
.rv{font-size:11px;font-weight:500;min-width:36px;text-align:right}
/* ASSET */
.arow{display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:0.5px solid var(--bd)}
.arow:last-child{border-bottom:none}
.ati{font-size:11px;font-weight:500;min-width:52px}
.an{font-size:10px;color:var(--t3);flex:1}
.ap{font-size:11px;min-width:70px;text-align:right}
.ac{font-size:11px;font-weight:500;min-width:52px;text-align:right}
.abadge{font-size:9px;padding:2px 7px;border-radius:4px;min-width:42px;text-align:center}
.bg{background:rgba(34,197,94,.1);color:var(--green)}
.br{background:rgba(239,68,68,.1);color:var(--red)}
.bx{background:rgba(148,163,184,.08);color:var(--t2)}
/* SEP */
.sep{border:none;border-top:0.5px solid var(--bd);margin:.8rem 0}
/* INDICATOR TABLE */
.indic{display:flex;flex-direction:column;gap:7px}
.irow{display:flex;justify-content:space-between;align-items:center;font-size:11px}
.ikey{color:var(--t2)}
.ival{font-weight:500}
/* REGIME */
.rcal{border-left:3px solid var(--green) }
.rstress{border-left:3px solid var(--amber)}
.rcrash{border-left:3px solid var(--red)  }
.rblock{background:var(--bg3);border-radius:6px;padding:10px 12px;margin-bottom:8px}
/* STATUS */
.status-bar{display:flex;align-items:center;gap:8px;background:var(--bg3);padding:8px 14px;border-radius:6px;font-size:10px;color:var(--t3)}
/* REFRESH BTN */
.refresh-btn{font-size:11px;padding:5px 14px;background:transparent;border:0.5px solid var(--bd2);border-radius:5px;color:var(--t2);cursor:pointer;transition:all .15s;font-family:var(--mono)}
.refresh-btn:hover{background:var(--bg3);color:var(--t)}
/* SCROLLBAR */
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:2px}
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div>
      <div class="logo-title">&#x1F4C8; S&P 500 ML Predictor</div>
      <div class="logo-sub">Intelligent Trading System · Dual-Regime Ensemble</div>
    </div>
    <div class="live-row">
      <span class="live-dot"></span>
      <span style="font-size:10px;color:var(--t2)">LIVE</span>
      <span class="badge-pill" id="clock">--:--:--</span>
      <span class="badge-pill" id="last-update">Chargement...</span>
      <button class="refresh-btn" onclick="loadData()">&#x21BB; Actualiser</button>
    </div>
  </div>

  <div class="status-bar" id="status-bar">
    <span>&#9679;</span>
    <span id="status-text">Connexion à SQLite...</span>
  </div>
  <div style="height:14px"></div>

  <!-- MÉTRIQUES -->
  <div class="g4">
    <div class="mc">
      <div class="ml">S&P 500</div>
      <div class="mv green" id="sp-prix">---</div>
      <div class="ms" id="sp-var">---</div>
    </div>
    <div class="mc">
      <div class="ml">VIX</div>
      <div class="mv" id="vix-val" style="color:var(--blue)">---</div>
      <div class="ms" id="vix-regime">---</div>
    </div>
    <div class="mc">
      <div class="ml">P(hausse 5j)</div>
      <div class="mv" id="proba-val">---</div>
      <div class="ms" id="proba-sub">RF · XGB · LR ensemble</div>
    </div>
    <div class="mc">
      <div class="ml">Exposition cible</div>
      <div class="mv" id="expo-val">---</div>
      <div class="ms" id="expo-sub">allocation dynamique</div>
    </div>
  </div>

  <!-- SIGNAL + TENDANCE + ASSETS -->
  <div class="g21">

    <div class="card">
      <div class="ct">Signal de trading</div>
      <div class="sig-hero">
        <div class="sig-tag sig-gray" id="signal-badge">--- SIGNAL ---</div>
        <div style="font-size:10px;color:var(--t3)" id="signal-meta">Chargement du modèle...</div>
        <div class="pbar"><div class="pfill" id="proba-fill" style="width:50%;background:var(--t3)"></div></div>
        <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--t3)">
          <span>VENTE &larr;</span><span>&rarr; ACHAT</span>
        </div>
      </div>
      <div class="sep"></div>
      <div class="ct">Tendance multi-horizons</div>
      <div class="row"><div class="rl">Court terme 5j</div><div class="rb"><div class="rf" id="bar-ct" style="width:0%;background:var(--green)"></div></div><div class="rv" id="val-ct">---</div></div>
      <div class="row"><div class="rl">Moyen terme 20j</div><div class="rb"><div class="rf" id="bar-mt" style="width:0%;background:var(--green)"></div></div><div class="rv" id="val-mt">---</div></div>
      <div class="row"><div class="rl">Long terme 60j</div><div class="rb"><div class="rf" id="bar-lt" style="width:0%;background:var(--green)"></div></div><div class="rv" id="val-lt">---</div></div>
      <div class="row"><div class="rl">Modèle ML</div><div class="rb"><div class="rf" id="bar-ml" style="width:0%;background:var(--blue)"></div></div><div class="rv" id="val-ml">---</div></div>
      <div class="sep"></div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
        <div style="font-size:11px"><span style="color:var(--t3)">MA20</span><br><span id="ma20">---</span></div>
        <div style="font-size:11px"><span style="color:var(--t3)">MA50</span><br><span id="ma50">---</span></div>
        <div style="font-size:11px"><span style="color:var(--t3)">MA200</span><br><span id="ma200">---</span></div>
        <div style="font-size:11px"><span style="color:var(--t3)">Support</span><br><span style="color:var(--amber)" id="support">---</span></div>
        <div style="font-size:11px"><span style="color:var(--t3)">Résistance</span><br><span style="color:var(--yellow)" id="resistance">---</span></div>
        <div style="font-size:11px"><span style="color:var(--t3)">RSI 14</span><br><span id="rsi-val">---</span></div>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:11px">
      <div class="card">
        <div class="ct">Multi-actifs</div>
        <div id="assets-box">
          <div style="font-size:10px;color:var(--t3)">Chargement...</div>
        </div>
      </div>
      <div class="card">
        <div class="ct">Indicateurs techniques</div>
        <div class="indic" id="indic-box">
          <div class="irow"><div class="ikey">MACD hist.</div><div class="ival" id="macd-val">---</div></div>
          <div class="irow"><div class="ikey">Bollinger Z</div><div class="ival" id="bb-val">---</div></div>
          <div class="irow"><div class="ikey">Momentum 5j</div><div class="ival" id="mom-val">---</div></div>
          <div class="irow"><div class="ikey">Écart MA20</div><div class="ival" id="ema20-val">---</div></div>
          <div class="irow"><div class="ikey">Écart MA50</div><div class="ival" id="ema50-val">---</div></div>
          <div class="irow"><div class="ikey">Écart MA200</div><div class="ival" id="ema200-val">---</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- CHART PERFORMANCE -->
  <div class="g2">
    <div class="card">
      <div class="ct">Performance cumulée — Régime CALME</div>
      <div style="display:flex;gap:14px;margin-bottom:10px" id="legend-calme"></div>
      <div style="position:relative;height:190px"><canvas id="chartCalme" role="img" aria-label="Performance cumulée régime calme">Courbe de performance ML vs Buy and Hold en régime calme.</canvas></div>
    </div>
    <div class="card">
      <div class="ct">Performance cumulée — Régime STRESS</div>
      <div style="display:flex;gap:14px;margin-bottom:10px" id="legend-stress"></div>
      <div style="position:relative;height:190px"><canvas id="chartStress" role="img" aria-label="Performance cumulée régime stress">Courbe de performance ML vs Buy and Hold en régime stress.</canvas></div>
    </div>
  </div>

  <!-- RÉGIMES + HISTORIQUE -->
  <div class="g32">
    <div class="card">
      <div class="ct">Historique des signaux de tendance (30 derniers jours)</div>
      <div id="hist-box" style="overflow-x:auto">
        <div style="font-size:10px;color:var(--t3)">Chargement...</div>
      </div>
    </div>
    <div class="card">
      <div class="ct">Analyse par régime VIX</div>
      <div class="rblock rcal">
        <div style="font-size:11px;font-weight:500;color:var(--green);margin-bottom:4px">Très calme — VIX &lt; 15</div>
        <div style="font-size:10px;color:var(--t2)">Hit Rate <b>61.1%</b> · Sharpe <b>5.07</b> · 1062 j.</div>
      </div>
      <div class="rblock rcal" style="border-left-color:var(--blue)">
        <div style="font-size:11px;font-weight:500;color:var(--blue);margin-bottom:4px">Calme — VIX 15-20</div>
        <div style="font-size:10px;color:var(--t2)">Hit Rate <b>55.0%</b> · Sharpe <b>2.29</b> · 962 j.</div>
      </div>
      <div class="rblock rstress">
        <div style="font-size:11px;font-weight:500;color:var(--amber);margin-bottom:4px">Volatil — VIX 20-30</div>
        <div style="font-size:10px;color:var(--t2)">Hit Rate <b>46.4%</b> · Sharpe <b>-0.96</b> · 705 j.</div>
      </div>
      <div class="rblock rcrash">
        <div style="font-size:11px;font-weight:500;color:var(--red);margin-bottom:4px">Extrême — VIX &gt; 30</div>
        <div style="font-size:10px;color:var(--t2)">Hit Rate <b>37.3%</b> · Sharpe <b>-2.62</b> · 158 j.</div>
      </div>
      <div class="sep"></div>
      <div style="font-size:10px;color:var(--t3)">Corrélation SP500/VIX : <span style="color:var(--red);font-weight:500">-0.727</span></div>
    </div>
  </div>

  <div style="border-top:0.5px solid var(--bd);padding-top:12px;display:flex;justify-content:space-between">
    <div style="font-size:9px;color:var(--t3)">S&P 500 ML Predictor · RF + XGBoost + LR · Walk-forward validation · Zero data leakage</div>
    <div style="font-size:9px;color:var(--t3)">Actualisation auto toutes les 60s · <span id="next-refresh">60</span>s</div>
  </div>

</div>

<script>
let chartCalme  = null;
let chartStress = null;
let countdown   = 60;
let countTimer  = null;

function fmt(n, d=2) { return parseFloat(n).toFixed(d); }
function fmtPrix(n)  { return parseFloat(n).toLocaleString('fr-FR', {minimumFractionDigits:2, maximumFractionDigits:2}); }

function tick() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('fr-FR');
}
tick();
setInterval(tick, 1000);

function startCountdown() {
  countdown = 60;
  if(countTimer) clearInterval(countTimer);
  countTimer = setInterval(() => {
    countdown--;
    const el = document.getElementById('next-refresh');
    if(el) el.textContent = countdown;
    if(countdown <= 0) loadData();
  }, 1000);
}

function setStatus(ok, msg) {
  const bar = document.getElementById('status-bar');
  const txt = document.getElementById('status-text');
  if(txt) txt.textContent = msg;
  if(bar) bar.style.borderLeft = ok ? '3px solid var(--green)' : '3px solid var(--red)';
}

function sigClass(couleur) {
  return {green:'sig-green', red:'sig-red', amber:'sig-amber', gray:'sig-gray'}[couleur] || 'sig-gray';
}

function barColor(score) {
  return score >= 0 ? 'var(--green)' : 'var(--red)';
}

function buildLegend(id, items) {
  const el = document.getElementById(id);
  if(!el) return;
  el.innerHTML = items.map(i =>
    `<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#94a3b8">
      <span style="width:10px;height:3px;background:${i.c};border-radius:2px;display:inline-block"></span>
      ${i.label}
    </span>`).join('');
}

function buildChart(id, labels, datasets) {
  const ctx = document.getElementById(id);
  if(!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index', intersect: false,
          backgroundColor: '#0d1320', borderColor: '#1e2d45', borderWidth: 1,
          titleColor: '#94a3b8', bodyColor: '#e2e8f0',
          callbacks: { label: c => ` ${c.dataset.label}: ${parseFloat(c.parsed.y).toFixed(1)}%` }
        }
      },
      scales: {
        x: { grid:{color:'rgba(148,163,184,.07)'}, ticks:{color:'#4a5568',font:{size:9},maxRotation:0,maxTicksLimit:6} },
        y: { grid:{color:'rgba(148,163,184,.07)'}, ticks:{color:'#4a5568',font:{size:9},callback:v=>v.toFixed(0)+'%'} }
      },
      interaction: { mode:'index', intersect:false }
    }
  });
}

function updateChart(chart, labels, datasets) {
  if(!chart) return;
  chart.data.labels = labels;
  chart.data.datasets = datasets;
  chart.update('none');
}

async function loadData() {
  setStatus(true, 'Actualisation...');
  startCountdown();
  try {
    const res  = await fetch('/api/data');
    const data = await res.json();

    document.getElementById('last-update').textContent = data.timestamp;

    // ── MARCHÉ ──
    const sp = data.marche?.sp500 || {};
    const vx = data.marche?.vix  || {};

    const spEl = document.getElementById('sp-prix');
    if(spEl) {
      spEl.textContent = fmtPrix(sp.prix || 0);
      spEl.className   = 'mv ' + ((sp.variation||0) >= 0 ? 'green' : 'red');
    }
    const spVar = document.getElementById('sp-var');
    if(spVar) {
      const v = sp.variation || 0;
      spVar.innerHTML = `<span class="${v>=0?'green':'red'}">${v>=0?'▲':'▼'} ${fmt(Math.abs(v))}%</span>&nbsp;aujourd'hui`;
    }

    const vixEl = document.getElementById('vix-val');
    if(vixEl) vixEl.textContent = fmt(vx.prix || 0, 1);

    // ── SIGNAL ──
    const sig = data.signal || {};
    const regime = sig.regime || 'calme';
    const regColors = {calme:'#3b82f6', stress:'#f97316', crash:'#ef4444'};
    const regEl = document.getElementById('vix-regime');
    if(regEl) regEl.innerHTML =
      `<span style="background:${regColors[regime]}22;color:${regColors[regime]};font-size:9px;padding:2px 7px;border-radius:4px;text-transform:uppercase">${regime}</span>`;

    const probaEl = document.getElementById('proba-val');
    if(probaEl) {
      probaEl.textContent = (sig.proba || 50) + '%';
      probaEl.className   = 'mv ' + ((sig.proba||50) >= 55 ? 'green' : (sig.proba||50) < 45 ? 'red' : 'amber');
    }

    const expoEl = document.getElementById('expo-val');
    if(expoEl) {
      expoEl.textContent = (sig.exposition || 0) + '%';
      expoEl.className   = 'mv ' + ((sig.exposition||0) >= 50 ? 'green' : 'amber');
    }
    const expoSub = document.getElementById('expo-sub');
    if(expoSub) expoSub.textContent = sig.signal || '---';

    const sbadge = document.getElementById('signal-badge');
    if(sbadge) {
      const icons = {green:'▲', red:'▼', amber:'◆', gray:'→'};
      sbadge.textContent = (icons[sig.couleur]||'→') + ' ' + (sig.signal||'---');
      sbadge.className   = 'sig-tag sig-' + (sig.couleur||'gray');
    }
    const smeta = document.getElementById('signal-meta');
    if(smeta) smeta.textContent =
      `Conviction ${sig.conviction||'---'} · Score ${sig.score||0}/100 · Régime ${regime.toUpperCase()}`;

    const pfill = document.getElementById('proba-fill');
    if(pfill) {
      const p = Math.min(100, Math.max(0, sig.proba||50));
      pfill.style.width      = p + '%';
      pfill.style.background = p >= 60 ? 'var(--green)' : p < 40 ? 'var(--red)' : 'var(--amber)';
    }

    // ── TENDANCE ──
    const t = data.tendance || {};
    const scores = {ct: t.score_ct||0, mt: t.score_mt||0, lt: t.score_lt||0};
    const mlScore = ((sig.proba||50) - 50) * 2;

    [['ct',scores.ct], ['mt',scores.mt], ['lt',scores.lt], ['ml',mlScore]].forEach(([id, sc]) => {
      const bar = document.getElementById('bar-'+id);
      const val = document.getElementById('val-'+id);
      if(bar) {
        bar.style.width      = Math.min(100, Math.abs(sc)) + '%';
        bar.style.background = sc >= 0 ? 'var(--green)' : 'var(--red)';
      }
      if(val) {
        val.textContent = (sc >= 0 ? '+' : '') + fmt(sc, 0);
        val.className   = 'rv ' + (sc >= 0 ? 'green' : 'red');
      }
    });

    const setTxt = (id, val) => { const el=document.getElementById(id); if(el) el.textContent=val; };

    setTxt('ma20',      fmtPrix(t.ma20||0));
    setTxt('ma50',      fmtPrix(t.ma50||0));
    setTxt('ma200',     fmtPrix(t.ma200||0));
    setTxt('support',   fmtPrix(t.support_1||0));
    setTxt('resistance',fmtPrix(t.resistance||0));
    setTxt('rsi-val',   fmt(t.rsi||0,1));
    setTxt('macd-val',  fmt(t.macd_hist||0,4));
    setTxt('bb-val',    fmt(t.bb_zscore||0,2));
    setTxt('mom-val',   (t.mom5||0)>=0 ? '+'+fmt(t.mom5||0,2)+'%' : fmt(t.mom5||0,2)+'%');
    setTxt('ema20-val', (t.ecart_ma20||0)>=0 ? '+'+fmt(t.ecart_ma20||0,2)+'%' : fmt(t.ecart_ma20||0,2)+'%');
    setTxt('ema50-val', (t.ecart_ma50||0)>=0 ? '+'+fmt(t.ecart_ma50||0,2)+'%' : fmt(t.ecart_ma50||0,2)+'%');
    setTxt('ema200-val',(t.ecart_ma200||0)>=0? '+'+fmt(t.ecart_ma200||0,2)+'%':fmt(t.ecart_ma200||0,2)+'%');

    // ── ASSETS ──
    const sigMap = (v) => v>=1?'ACHAT':v<=-0.5?'VENTE':'NEU';
    const clsMap = (v) => v>=0?'bg':v<-0.5?'br':'bx';
    const assetBox = document.getElementById('assets-box');
    if(assetBox) {
      const actifs = ['sp500','vix','bitcoin','gold','dxy'];
      assetBox.innerHTML = actifs.map(nom => {
        const a = data.marche?.[nom] || {};
        const v = a.variation || 0;
        const cls = clsMap(v);
        const badge = nom==='vix'?(v<0?'<span class="abadge bg">BON</span>':'<span class="abadge bx">NEU</span>'):
                      `<span class="abadge ${cls}">${v>=1?'ACHAT':v<=-1?'VENTE':'NEU'}</span>`;
        return `<div class="arow">
          <div class="ati" style="color:var(--blue)">${a.ticker||nom.toUpperCase()}</div>
          <div class="an">${a.label||nom}</div>
          <div class="ap">${fmtPrix(a.prix||0)}</div>
          <div class="ac ${v>=0?'green':'red'}">${v>=0?'▲':'▼'} ${fmt(Math.abs(v),2)}%</div>
          ${badge}
        </div>`;
      }).join('');
    }

    // ── CHARTS PERF ──
    const perf = data.perf || {};
    const dsCalme = [
      { label:'ML Alloc.', data:(perf.calme?.ml||[]),  borderColor:'#22c55e', backgroundColor:'rgba(34,197,94,.05)', fill:true, tension:.4, pointRadius:0, borderWidth:2 },
      { label:'Buy&Hold',  data:(perf.calme?.bh||[]),  borderColor:'#4a5568', backgroundColor:'transparent', fill:false, tension:.4, pointRadius:0, borderWidth:1.5, borderDash:[5,3] },
      { label:'Binaire',   data:(perf.calme?.bin||[]), borderColor:'#3b82f6', backgroundColor:'transparent', fill:false, tension:.4, pointRadius:0, borderWidth:1.5 },
    ];
    const dsStress = [
      { label:'ML Alloc.', data:(perf.stress?.ml||[]),  borderColor:'#22c55e', backgroundColor:'rgba(34,197,94,.05)', fill:true, tension:.4, pointRadius:0, borderWidth:2 },
      { label:'Buy&Hold',  data:(perf.stress?.bh||[]),  borderColor:'#4a5568', backgroundColor:'transparent', fill:false, tension:.4, pointRadius:0, borderWidth:1.5, borderDash:[5,3] },
      { label:'Binaire',   data:(perf.stress?.bin||[]), borderColor:'#3b82f6', backgroundColor:'transparent', fill:false, tension:.4, pointRadius:0, borderWidth:1.5 },
    ];

    buildLegend('legend-calme', [{c:'#22c55e',label:'ML Alloc.'},{c:'#4a5568',label:'Buy&Hold'},{c:'#3b82f6',label:'Binaire'}]);
    buildLegend('legend-stress',[{c:'#22c55e',label:'ML Alloc.'},{c:'#4a5568',label:'Buy&Hold'},{c:'#3b82f6',label:'Binaire'}]);

    if(!chartCalme) chartCalme = buildChart('chartCalme', perf.calme?.dates||[], dsCalme);
    else updateChart(chartCalme, perf.calme?.dates||[], dsCalme);

    if(!chartStress) chartStress = buildChart('chartStress', perf.stress?.dates||[], dsStress);
    else updateChart(chartStress, perf.stress?.dates||[], dsStress);

    // ── HISTORIQUE TENDANCES ──
    const hist = data.historique || [];
    const histBox = document.getElementById('hist-box');
    if(histBox && hist.length > 0) {
      const cols = ['date','score_global','score_ct','score_mt','score_lt','tendance_global','conviction','vix','regime'];
      histBox.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:10px">
        <thead><tr>${['Date','Score','CT','MT','LT','Tendance','Conv.','VIX','Régime'].map(h=>
          `<th style="padding:5px 8px;text-align:left;color:var(--t3);border-bottom:0.5px solid var(--bd);font-weight:500">${h}</th>`).join('')}</tr></thead>
        <tbody>${[...hist].reverse().map(row => {
          const sc = parseFloat(row.score_global||0);
          const col = sc>20?'var(--green)':sc<-20?'var(--red)':'var(--amber)';
          return `<tr style="border-bottom:0.5px solid var(--bd)">
            <td style="padding:5px 8px;color:var(--t2)">${(row.date||'').substring(0,10)}</td>
            <td style="padding:5px 8px;font-weight:500;color:${col}">${sc>=0?'+':''}${sc.toFixed(1)}</td>
            <td style="padding:5px 8px;color:${parseFloat(row.score_ct||0)>=0?'var(--green)':'var(--red)'}">${parseFloat(row.score_ct||0).toFixed(0)}</td>
            <td style="padding:5px 8px;color:${parseFloat(row.score_mt||0)>=0?'var(--green)':'var(--red)'}">${parseFloat(row.score_mt||0).toFixed(0)}</td>
            <td style="padding:5px 8px;color:${parseFloat(row.score_lt||0)>=0?'var(--green)':'var(--red)'}">${parseFloat(row.score_lt||0).toFixed(0)}</td>
            <td style="padding:5px 8px;color:var(--t2);max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${(row.tendance_global||'---').replace(/[▲▽▼△]/g,'').trim()}</td>
            <td style="padding:5px 8px">${row.conviction||'---'}</td>
            <td style="padding:5px 8px">${parseFloat(row.vix||0).toFixed(1)}</td>
            <td style="padding:5px 8px;text-transform:capitalize">${row.regime||'---'}</td>
          </tr>`;}).join('')}</tbody></table>`;
    } else if(histBox) {
      histBox.innerHTML = '<div style="font-size:10px;color:var(--t3)">Aucun historique — lance 4_prediction_tendance.py</div>';
    }

    setStatus(true, `Données chargées · ${data.timestamp} · Modèle: ${sig.modele_ok ? 'OK ✓' : 'Absent — lance 2_train_models_FINAL.py'}`);

  } catch(err) {
    setStatus(false, 'Erreur connexion : ' + err.message);
    console.error(err);
  }
}

loadData();
setInterval(loadData, 60000);
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


# ─────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  S&P 500 ML PREDICTOR — DASHBOARD LIVE")
    print("=" * 55)
    print(f"  Base SQLite : {DB_PATH}")
    print(f"  URL         : http://localhost:5000")
    print(f"  Actualisation auto : toutes les 60 secondes")
    print("=" * 55)
    print("  Ctrl+C pour arrêter")
    print()

    if not os.path.exists(DB_PATH):
        print(f"  ATTENTION : {DB_PATH} introuvable.")
        print("  Lance d'abord : python 01_init_data.py")

    app.run(debug=False, port=5000, host='0.0.0.0')