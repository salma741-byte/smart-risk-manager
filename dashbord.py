# ============================================================
#  dashboard_v2.py  —  S&P 500 ML Predictor · Live Dashboard
#  Version propre — corrige tous les bugs précédents
#
#  Installation : pip install flask
#  Lancement    : python dashboard_v2.py
#  URL          : http://localhost:5000
# ============================================================

import os, json, sqlite3, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, render_template_string

app     = Flask(__name__)
DB_PATH = "data/market_data.db"

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def db():
    return sqlite3.connect(DB_PATH)

def f(v, d=2):
    try:    return round(float(v), d)
    except: return 0.0

# ─────────────────────────────────────────────────────────────
# 1. DONNÉES MARCHÉ
# ─────────────────────────────────────────────────────────────

def marche():
    actifs = {
        'sp500':   ('^GSPC',    'S&P 500'),
        'vix':     ('^VIX',     'VIX'),
        'bitcoin': ('BTC-USD',  'Bitcoin'),
        'gold':    ('GC=F',     'Or'),
        'dxy':     ('DX-Y.NYB', 'Dollar'),
    }
    out = {}
    try:
        conn = db()
        for nom, (ticker, label) in actifs.items():
            try:
                rows = pd.read_sql(
                    f"SELECT close FROM {nom}_prices ORDER BY date DESC LIMIT 2",
                    conn)
                p0 = f(rows['close'].iloc[0]) if len(rows) > 0 else 0
                p1 = f(rows['close'].iloc[1]) if len(rows) > 1 else p0
                out[nom] = {
                    'ticker': ticker, 'label': label,
                    'prix': p0,
                    'var':  f((p0 - p1) / p1 * 100) if p1 else 0,
                }
            except:
                out[nom] = {'ticker': ticker, 'label': label, 'prix': 0, 'var': 0}
        conn.close()
    except Exception as e:
        print(f"Erreur marché: {e}")
    return out


# ─────────────────────────────────────────────────────────────
# 2. SIGNAL ML
# ─────────────────────────────────────────────────────────────

def signal_ml():
    DEFAULT = {'vix': 0, 'regime': 'calme', 'proba': 50,
               'expo': 60, 'signal': 'N/A', 'couleur': 'gray',
               'score': 0, 'conviction': 'FAIBLE', 'ok': False, 'date': '---'}
    try:
        conn = db()
        df   = pd.read_sql("SELECT * FROM sp500_ml_features ORDER BY date DESC LIMIT 300",
                           conn, parse_dates=['date'])
        conn.close()

        if df.empty:
            return DEFAULT

        row = df.iloc[[0]].copy()

        # VIX et régime
        vix_val = f(row['vix_close'].iloc[0]) if 'vix_close' in row.columns else 20.0
        regime  = 'crash' if vix_val >= 30 else 'stress' if vix_val >= 20 else 'calme'

        # Enrichissement — delta_vix, rolling_corr, zscore_60d
        if 'vix_close' in df.columns and len(df) >= 2:
            row['delta_vix']           = float(df['vix_close'].iloc[0] - df['vix_close'].iloc[1])
            row['rolling_corr_sp_vix'] = float(df['ret_1d'].corr(df['vix_ret_1d'])) if 'vix_ret_1d' in df.columns else 0.0
            row['vix_mean_reversion']  = 0.0
            row['vix_pct_rank']        = float((df['vix_close'] <= vix_val).mean())
        else:
            for c in ['delta_vix','rolling_corr_sp_vix','vix_mean_reversion','vix_pct_rank']:
                row[c] = 0.0

        if 'close' in df.columns and len(df) >= 60:
            mu = df['close'].iloc[:60].mean()
            sd = df['close'].iloc[:60].std()
            row['zscore_price_60d'] = float((df['close'].iloc[0] - mu) / sd) if sd else 0.0
        else:
            row['zscore_price_60d'] = 0.0

        for c in ['drawdown_20d', 'drawdown_50d']:
            if c not in row.columns:
                row[c] = 0.0

        row.replace([np.inf, -np.inf], np.nan, inplace=True)
        row.fillna(0, inplace=True)

        # Modèle
        modeles  = joblib.load(f"models/ensemble_{regime}.pkl")
        scaler   = joblib.load(f"models/scaler_{regime}.pkl")
        features = joblib.load(f"models/features_{regime}.pkl")

        feats = [x for x in features if x in row.columns]
        X     = row[feats].values.astype(np.float64)
        Xs    = scaler.transform(X)

        poids = {'rf': 0.30, 'xgb': 0.50, 'lr': 0.20}
        proba = sum(m['model'].predict_proba(Xs)[0, 1] * poids.get(n, 0.33)
                    for n, m in modeles.items())

        # Signal
        if regime == 'calme':
            if   proba >= 0.65: expo=100; sig='FORT ACHAT'; col='green'
            elif proba >= 0.60: expo=80;  sig='ACHAT';      col='green'
            elif proba >= 0.55: expo=60;  sig='ACHAT';      col='green'
            elif proba >= 0.50: expo=40;  sig='REDUIT';     col='amber'
            elif proba >= 0.44: expo=30;  sig='REDUIT';     col='amber'
            else:               expo=0;   sig='NEUTRE';     col='gray'
        elif regime == 'stress':
            if   proba >= 0.58: expo=80;  sig='ACHAT';      col='green'
            elif proba >= 0.52: expo=50;  sig='ACHAT';      col='green'
            elif proba >= 0.45: expo=30;  sig='REDUIT';     col='amber'
            elif proba >= 0.38: expo=0;   sig='NEUTRE';     col='gray'
            else:               expo=-20; sig='SHORT';      col='red'
        else:
            if   proba >= 0.65: expo=50;  sig='ACHAT TIMIDE';col='amber'
            elif proba >= 0.55: expo=20;  sig='REDUIT';      col='amber'
            else:               expo=0;   sig='NEUTRE';      col='gray'

        score = (proba - 0.5) * 200
        conv  = 'FORTE' if abs(score) >= 35 else 'MODEREE' if abs(score) >= 15 else 'FAIBLE'
        date  = str(df['date'].iloc[0])[:10]

        return {'vix': vix_val, 'regime': regime,
                'proba': f(proba*100,1), 'expo': expo,
                'signal': sig, 'couleur': col,
                'score': f(score,1), 'conviction': conv,
                'ok': True, 'date': date}

    except FileNotFoundError:
        return {**DEFAULT, 'ok': False}
    except Exception as e:
        print(f"Erreur ML: {e}")
        return {**DEFAULT, 'ok': False}


# ─────────────────────────────────────────────────────────────
# 3. TENDANCE TECHNIQUE
# ─────────────────────────────────────────────────────────────

def tendance():
    try:
        conn = db()
        sp   = pd.read_sql("SELECT close,high,low,open,volume FROM sp500_prices ORDER BY date DESC LIMIT 250",
                           conn).iloc[::-1].reset_index(drop=True)
        conn.close()

        c = sp['close']
        ma20  = f(c.rolling(20).mean().iloc[-1])
        ma50  = f(c.rolling(50).mean().iloc[-1])
        ma200 = f(c.rolling(200).mean().iloc[-1])
        prix  = f(c.iloc[-1])

        delta = c.diff()
        rsi   = f(100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean() /
                                   (-delta.clip(upper=0)).rolling(14).mean())).iloc[-1])

        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd_hist = f((ema12 - ema26 - (ema12-ema26).ewm(span=9).mean()).iloc[-1], 4)
        bb_z      = f(((c - c.rolling(20).mean()) / c.rolling(20).std()).iloc[-1], 2)
        mom5      = f((c.iloc[-1]/c.iloc[-6]-1)*100, 2)

        def sc(v, seuils):  # score discret
            for seuil, pts in seuils:
                if v >= seuil: return pts
            return seuils[-1][1]

        s_rsi  = sc(rsi,  [(60,20),(50,10),(40,-10),(0,-20)])
        s_macd = 20 if macd_hist > 0 else -20
        s_mom5 = sc(mom5, [(2,20),(0,10),(-2,-10),(-99,-20)])
        s_vix  = 0  # calculé dans signal_ml

        s_ma20  = sc((prix/ma20-1)*100,  [(3,20),(0,10),(-3,-10),(-99,-20)])
        s_ma50  = sc((prix/ma50-1)*100,  [(5,20),(0,10),(-5,-10),(-99,-20)])
        s_cross = 20 if ma20 > ma50 else -20
        s_bb    = sc(bb_z, [(1.5,20),(0.5,10),(-0.5,-10),(-99,-20)])

        s_ma200 = sc((prix/ma200-1)*100, [(5,30),(0,15),(-5,-15),(-99,-30)])
        s_lt_c  = 15 if ma50 > ma200 else -15
        s_lt_m  = sc(mom5, [(2,20),(0,10),(-99,-10)])

        s_ct = s_rsi + s_macd + s_mom5
        s_mt = s_ma20 + s_ma50 + s_cross + s_bb
        s_lt = s_ma200 + s_lt_c + s_lt_m

        norm = lambda s, mx: round(s/mx*100, 1)

        return {
            'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
            'rsi': rsi, 'macd_hist': macd_hist, 'bb_z': bb_z, 'mom5': mom5,
            'score_ct': norm(s_ct, 60), 'score_mt': norm(s_mt, 80), 'score_lt': norm(s_lt, 60),
            'ecart20':  f((prix/ma20-1)*100,2),
            'ecart50':  f((prix/ma50-1)*100,2),
            'ecart200': f((prix/ma200-1)*100,2),
            'resistance': f(c.rolling(20).max().iloc[-1]*1.02),
            'support1':   f(ma50*0.97),
            'support2':   f(ma200*0.98),
        }
    except Exception as e:
        print(f"Erreur tendance: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# 4. PERFORMANCE — lit les CSV et détecte les colonnes auto
# ─────────────────────────────────────────────────────────────

def perf():
    out = {}
    for regime in ['calme', 'stress']:
        path = f"results/comparaison_{regime}.csv"
        if not os.path.exists(path):
            out[regime] = {}
            continue
        try:
            df = pd.read_csv(path)
            print(f"  [perf] {regime} colonnes: {df.columns.tolist()}")

            # Détecter colonnes dates
            date_col = next((c for c in df.columns
                             if 'date' in c.lower()), None)

            # Détecter colonnes cumul (plusieurs noms possibles)
            def find_col(keywords):
                for kw in keywords:
                    for c in df.columns:
                        if kw in c.lower():
                            return c
                return None

            col_ml  = find_col(['cumul_strat','cumul_dyn','strat','dyn'])
            col_bh  = find_col(['cumul_bh','bh'])
            col_bin = find_col(['cumul_bin','bin'])

            # Si cumul absent, recalculer depuis ret
            if col_ml is None:
                ret_col = find_col(['ret_strat','ret_dyn','ret_strategie'])
                if ret_col:
                    df['_cumul_ml'] = (1 + df[ret_col]).cumprod()
                    col_ml = '_cumul_ml'

            if col_bh is None:
                ret_bh = find_col(['ret_bh','ret_j1'])
                if ret_bh:
                    df['_cumul_bh'] = (1 + df[ret_bh]).cumprod()
                    col_bh = '_cumul_bh'

            if col_ml is None or col_bh is None:
                print(f"  [perf] colonnes ML/BH introuvables pour {regime}")
                out[regime] = {}
                continue

            df = df.dropna(subset=[col_ml, col_bh]).reset_index(drop=True)
            if len(df) == 0:
                out[regime] = {}
                continue

            # Convertir en % relatif à la première valeur
            def to_pct(col):
                s = df[col].values.astype(float)
                return ((s / s[0]) * 100 - 100).tolist()

            step = max(1, len(df) // 60)
            idx  = list(range(0, len(df), step))

            out[regime] = {
                'dates': [str(df[date_col].iloc[i])[:10] for i in idx] if date_col else [str(i) for i in idx],
                'ml':    [round(to_pct(col_ml)[i],  1) for i in idx],
                'bh':    [round(to_pct(col_bh)[i],  1) for i in idx],
                'bin':   [round(to_pct(col_bin)[i], 1) for i in idx] if col_bin else [],
            }
        except Exception as e:
            print(f"  [perf] Erreur {regime}: {e}")
            out[regime] = {}
    return out


# ─────────────────────────────────────────────────────────────
# 5. HISTORIQUE TENDANCES
# ─────────────────────────────────────────────────────────────

def historique():
    path = "results/historique_tendances.csv"
    if not os.path.exists(path):
        return []
    try:
        return pd.read_csv(path).tail(30).to_dict(orient='records')
    except:
        return []


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────

@app.route('/api/data')
def api_data():
    return jsonify({
        'ts'         : datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'marche'     : marche(),
        'signal'     : signal_ml(),
        'tendance'   : tendance(),
        'perf'       : perf(),
        'historique' : historique(),
    })

@app.route('/api/signal')
def api_signal():
    return jsonify(signal_ml())


# ─────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────

PAGE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S&P 500 ML Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--s1:#0d1117;--s2:#161b22;--s3:#21262d;
  --bd:#30363d;--bd2:#3d444d;
  --t:#f0f6fc;--t2:#8b949e;--t3:#484f58;
  --g:#3fb950;--g2:#1a7f37;
  --r:#f85149;--r2:#b91c1c;
  --b:#58a6ff;--b2:#1d6fd8;
  --a:#d29922;--a2:#9e6a03;
  --p:#bc8cff;
  --mono:'JetBrains Mono',monospace;
  --sans:'Space Grotesk',sans-serif;
}
html,body{background:var(--bg);color:var(--t);font-family:var(--sans);min-height:100vh;font-size:14px;line-height:1.5}
.wrap{max-width:1480px;margin:0 auto;padding:0 20px 60px}

/* TOP */
.topbar{display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid var(--bd);margin-bottom:20px}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:34px;height:34px;background:linear-gradient(135deg,var(--b2),var(--b));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px}
.brand-name{font-size:15px;font-weight:600;letter-spacing:-.01em}
.brand-sub{font-size:10px;color:var(--t3);letter-spacing:.08em;text-transform:uppercase}
.top-right{display:flex;align-items:center;gap:8px}
.pill{font-size:11px;font-family:var(--mono);background:var(--s2);border:1px solid var(--bd);padding:4px 12px;border-radius:20px;color:var(--t2)}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--g);animation:pulse 1.8s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.7)}}

/* GRIDS */
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.g21{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:16px}
.g31{display:grid;grid-template-columns:3fr 1.4fr;gap:12px;margin-bottom:16px}

/* STAT CARD */
.sc{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:14px 16px;position:relative;overflow:hidden}
.sc::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(88,166,255,.2),transparent)}
.sc-label{font-size:10px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;font-family:var(--mono);margin-bottom:6px}
.sc-val{font-size:26px;font-weight:600;line-height:1;margin-bottom:4px;font-family:var(--mono)}
.sc-sub{font-size:11px;color:var(--t2);display:flex;align-items:center;gap:6px}

/* CARD */
.card{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:16px 18px}
.card-h{font-size:10px;font-family:var(--mono);color:var(--t3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between}
.card-h span{font-size:10px;background:var(--s2);padding:2px 8px;border-radius:4px;border:1px solid var(--bd);color:var(--t2)}

/* COLORS */
.g{color:var(--g)}.r{color:var(--r)}.b{color:var(--b)}.a{color:var(--a)}.p{color:var(--p)}

/* SIGNAL */
.sig-center{display:flex;flex-direction:column;align-items:center;padding:20px 0 14px;gap:8px}
.sig-tag{font-size:20px;font-weight:600;font-family:var(--mono);padding:10px 28px;border-radius:8px;letter-spacing:.02em;border:1px solid}
.sg{background:rgba(63,185,80,.1);color:var(--g);border-color:rgba(63,185,80,.3)}
.sr{background:rgba(248,81,73,.1);color:var(--r);border-color:rgba(248,81,73,.3)}
.sa{background:rgba(210,153,34,.1);color:var(--a);border-color:rgba(210,153,34,.3)}
.sx{background:var(--s2);color:var(--t2);border-color:var(--bd)}
.sig-meta{font-size:11px;font-family:var(--mono);color:var(--t3);text-align:center}
.pbar{width:100%;height:4px;background:var(--s3);border-radius:2px;overflow:hidden;margin:6px 0 3px}
.pf{height:100%;border-radius:2px;transition:width 1s ease}
.pbar-labels{display:flex;justify-content:space-between;font-size:9px;font-family:var(--mono);color:var(--t3)}

/* HORIZON BARS */
.hrow{display:flex;align-items:center;gap:10px;margin-bottom:9px}
.hlbl{font-size:10px;font-family:var(--mono);color:var(--t2);min-width:100px}
.hbar{flex:1;height:5px;background:var(--s3);border-radius:3px;overflow:hidden;position:relative}
.hfill{position:absolute;top:0;height:100%;border-radius:3px;transition:all .8s ease}
.hval{font-size:11px;font-family:var(--mono);font-weight:500;min-width:38px;text-align:right}

/* TABLE ASSETS */
.arow{display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--bd)}
.arow:last-child{border-bottom:none}
.atic{font-size:11px;font-weight:600;font-family:var(--mono);min-width:56px}
.an{font-size:10px;color:var(--t3);flex:1}
.ap{font-size:11px;font-family:var(--mono);min-width:74px;text-align:right}
.av{font-size:11px;font-family:var(--mono);font-weight:600;min-width:54px;text-align:right}
.abadge{font-size:9px;font-family:var(--mono);padding:2px 7px;border-radius:3px;min-width:40px;text-align:center;border:1px solid}
.bag{background:rgba(63,185,80,.1);color:var(--g);border-color:rgba(63,185,80,.25)}
.bar{background:rgba(248,81,73,.1);color:var(--r);border-color:rgba(248,81,73,.25)}
.bax{background:var(--s2);color:var(--t3);border-color:var(--bd)}

/* INDIC TABLE */
.irow{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--bd)}
.irow:last-child{border-bottom:none}
.ik{font-size:11px;color:var(--t2)}
.iv{font-size:11px;font-family:var(--mono);font-weight:500}

/* REGIME BLOCKS */
.rblock{border-radius:8px;padding:10px 12px;margin-bottom:8px;border-left:3px solid}
.rcalme{background:rgba(63,185,80,.06);border-left-color:var(--g)}
.rstress{background:rgba(210,153,34,.06);border-left-color:var(--a)}
.rcrash{background:rgba(248,81,73,.06);border-left-color:var(--r)}
.rcalme2{background:rgba(88,166,255,.06);border-left-color:var(--b)}
.rblock-title{font-size:11px;font-weight:600;font-family:var(--mono);margin-bottom:3px}
.rblock-sub{font-size:10px;color:var(--t2)}

/* FEAT BARS */
.frow{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.frk{font-size:9px;font-family:var(--mono);color:var(--t3);min-width:14px}
.fn{font-size:10px;color:var(--t2);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fb{flex:0 0 88px;height:4px;background:var(--s3);border-radius:2px;overflow:hidden}
.ff{height:100%;border-radius:2px}
.fp{font-size:9px;font-family:var(--mono);color:var(--t3);min-width:30px;text-align:right}

/* HIST TABLE */
.htable{width:100%;border-collapse:collapse;font-size:11px}
.htable th{padding:6px 10px;text-align:left;color:var(--t3);border-bottom:1px solid var(--bd);font-weight:500;font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:.06em}
.htable td{padding:6px 10px;border-bottom:1px solid var(--bd);font-family:var(--mono)}
.htable tr:last-child td{border-bottom:none}
.htable tr:hover td{background:var(--s2)}

/* TABS */
.tabs{display:flex;gap:6px;margin-bottom:12px}
.tab{font-size:10px;font-family:var(--mono);padding:4px 12px;background:transparent;border:1px solid var(--bd);border-radius:4px;color:var(--t2);cursor:pointer;transition:.15s}
.tab.on{background:var(--s2);color:var(--t);border-color:var(--bd2)}

.sep{border:none;border-top:1px solid var(--bd);margin:.9rem 0}
.muted{color:var(--t3)}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:2px}
</style>
</head>
<body>
<div class="wrap">

<!-- TOPBAR -->
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">📈</div>
    <div>
      <div class="brand-name">S&P 500 ML Predictor</div>
      <div class="brand-sub">Intelligent Trading System · Dual-Regime Ensemble</div>
    </div>
  </div>
  <div class="top-right">
    <span class="live-dot"></span>
    <span class="pill" id="clock">--:--:--</span>
    <span class="pill" id="ts">---</span>
    <button onclick="load()" style="font-size:11px;font-family:var(--mono);background:var(--s2);border:1px solid var(--bd);padding:5px 14px;border-radius:20px;color:var(--t2);cursor:pointer">↻ Refresh</button>
  </div>
</div>

<!-- KPI -->
<div class="g4">
  <div class="sc"><div class="sc-label">S&P 500</div><div class="sc-val g" id="k-sp">---</div><div class="sc-sub" id="k-spv">---</div></div>
  <div class="sc"><div class="sc-label">VIX</div><div class="sc-val b" id="k-vix">---</div><div class="sc-sub" id="k-reg">---</div></div>
  <div class="sc"><div class="sc-label">P(hausse 5j)</div><div class="sc-val" id="k-prob">---</div><div class="sc-sub muted">RF · XGB · LR</div></div>
  <div class="sc"><div class="sc-label">Exposition</div><div class="sc-val" id="k-expo">---</div><div class="sc-sub" id="k-sig">---</div></div>
</div>

<!-- ROW 2 : SIGNAL + ASSETS + INDICATEURS -->
<div class="g3">

  <!-- Signal -->
  <div class="card">
    <div class="card-h">Signal de trading <span id="sig-date">---</span></div>
    <div class="sig-center">
      <div class="sig-tag sx" id="sig-tag">— SIGNAL —</div>
      <div class="sig-meta" id="sig-meta">Chargement du modèle...</div>
    </div>
    <div class="pbar"><div class="pf" id="pf" style="width:50%;background:var(--t3)"></div></div>
    <div class="pbar-labels"><span>← VENTE</span><span>ACHAT →</span></div>
    <div class="sep"></div>
    <div class="card-h">Tendance multi-horizons</div>
    <div class="hrow"><div class="hlbl">Court terme 5j</div><div class="hbar"><div class="hfill" id="h-ct" style="width:0;left:50%;background:var(--g)"></div></div><div class="hval" id="v-ct">---</div></div>
    <div class="hrow"><div class="hlbl">Moyen terme 20j</div><div class="hbar"><div class="hfill" id="h-mt" style="width:0;left:50%;background:var(--g)"></div></div><div class="hval" id="v-mt">---</div></div>
    <div class="hrow"><div class="hlbl">Long terme 60j</div><div class="hbar"><div class="hfill" id="h-lt" style="width:0;left:50%;background:var(--g)"></div></div><div class="hval" id="v-lt">---</div></div>
    <div class="hrow"><div class="hlbl">Modèle ML</div><div class="hbar"><div class="hfill" id="h-ml" style="width:0;left:50%;background:var(--b)"></div></div><div class="hval b" id="v-ml">---</div></div>
    <div class="sep"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:11px;font-family:var(--mono)">
      <div><div class="muted">MA20</div><div id="ma20">---</div></div>
      <div><div class="muted">MA50</div><div id="ma50">---</div></div>
      <div><div class="muted">MA200</div><div id="ma200">---</div></div>
      <div><div class="muted">Support</div><div class="a" id="sup">---</div></div>
      <div><div class="muted">Résistance</div><div style="color:var(--p)" id="res">---</div></div>
      <div><div class="muted">RSI 14</div><div id="rsi">---</div></div>
    </div>
  </div>

  <!-- Assets -->
  <div class="card">
    <div class="card-h">Multi-actifs</div>
    <div id="assets">
      <div class="muted" style="font-size:11px">Chargement...</div>
    </div>
    <div class="sep"></div>
    <div class="card-h">Indicateurs techniques</div>
    <div>
      <div class="irow"><div class="ik">MACD hist.</div><div class="iv" id="i-macd">---</div></div>
      <div class="irow"><div class="ik">Bollinger Z-score</div><div class="iv" id="i-bb">---</div></div>
      <div class="irow"><div class="ik">Momentum 5j</div><div class="iv" id="i-mom">---</div></div>
      <div class="irow"><div class="ik">Écart vs MA20</div><div class="iv" id="i-e20">---</div></div>
      <div class="irow"><div class="ik">Écart vs MA50</div><div class="iv" id="i-e50">---</div></div>
      <div class="irow"><div class="ik">Écart vs MA200</div><div class="iv" id="i-e200">---</div></div>
    </div>
  </div>

  <!-- Régimes VIX -->
  <div class="card">
    <div class="card-h">Analyse par régime VIX</div>
    <div class="rblock rcalme"><div class="rblock-title g">Très calme — VIX &lt; 15</div><div class="rblock-sub">Hit Rate <b>61.1%</b> · Sharpe <b>5.07</b> · 1 062 j.</div></div>
    <div class="rblock rcalme2"><div class="rblock-title b">Calme — VIX 15-20</div><div class="rblock-sub">Hit Rate <b>55.0%</b> · Sharpe <b>2.29</b> · 962 j.</div></div>
    <div class="rblock rstress"><div class="rblock-title a">Volatil — VIX 20-30</div><div class="rblock-sub">Hit Rate <b>46.4%</b> · Sharpe <b>-0.96</b> · 705 j.</div></div>
    <div class="rblock rcrash"><div class="rblock-title r">Extrême — VIX &gt; 30</div><div class="rblock-sub">Hit Rate <b>37.3%</b> · Sharpe <b>-2.62</b> · 158 j.</div></div>
    <div class="sep"></div>
    <div class="card-h">Corrélations clés</div>
    <div class="irow"><div class="ik">SP500 / VIX</div><div class="iv r">-0.727</div></div>
    <div class="irow"><div class="ik">SP500 / BTC</div><div class="iv b">+0.238</div></div>
    <div class="irow"><div class="ik">SP500 / Gold</div><div class="iv">+0.040</div></div>
    <div class="irow"><div class="ik">Gold / DXY</div><div class="iv r">-0.382</div></div>
  </div>
</div>

<!-- PERFORMANCE CHARTS -->
<div class="g2">
  <div class="card">
    <div class="card-h">Performance cumulée — Régime CALME
      <span>ML +43% · B&H +53% · Binaire +41%</span>
    </div>
    <div style="display:flex;gap:14px;margin-bottom:10px" id="leg-calme"></div>
    <div style="height:190px;position:relative"><canvas id="ch-calme" role="img" aria-label="Performance cumulée régime calme: ML Alloc +43%, Buy and Hold +53%, Binaire +41%"></canvas></div>
  </div>
  <div class="card">
    <div class="card-h">Performance cumulée — Régime STRESS
      <span>ML +27% · B&H +27% · Binaire +27%</span>
    </div>
    <div style="display:flex;gap:14px;margin-bottom:10px" id="leg-stress"></div>
    <div style="height:190px;position:relative"><canvas id="ch-stress" role="img" aria-label="Performance cumulée régime stress: ML Alloc +27%, Buy and Hold +27%, MaxDD réduit de -15% à -5%"></canvas></div>
  </div>
</div>

<!-- SHAP + HISTORIQUE -->
<div class="g31">
  <div class="card">
    <div class="card-h">Historique des signaux de tendance
      <span id="hist-count">--- entrées</span>
    </div>
    <div style="overflow-x:auto" id="hist-box">
      <div class="muted" style="font-size:11px">Chargement...</div>
    </div>
  </div>
  <div class="card">
    <div class="card-h">SHAP — Feature importance</div>
    <div class="tabs">
      <button class="tab on" onclick="shap('calme',this)">Calme</button>
      <button class="tab"    onclick="shap('stress',this)">Stress</button>
    </div>
    <div id="shap-calme">
      <div class="frow"><div class="frk">1</div><div class="fn">MACD Histogram</div><div class="fb"><div class="ff" style="width:100%;background:var(--b)"></div></div><div class="fp">0.069</div></div>
      <div class="frow"><div class="frk">2</div><div class="fn">Fear &amp; Greed Index</div><div class="fb"><div class="ff" style="width:98%;background:var(--b)"></div></div><div class="fp">0.068</div></div>
      <div class="frow"><div class="frk">3</div><div class="fn">Bollinger Z-score</div><div class="fb"><div class="ff" style="width:93%;background:var(--b)"></div></div><div class="fp">0.064</div></div>
      <div class="frow"><div class="frk">4</div><div class="fn">VIX Percentile 252j</div><div class="fb"><div class="ff" style="width:80%;background:var(--p)"></div></div><div class="fp">0.055</div></div>
      <div class="frow"><div class="frk">5</div><div class="fn">ADX — Force tendance</div><div class="fb"><div class="ff" style="width:74%;background:var(--p)"></div></div><div class="fp">0.051</div></div>
      <div class="frow"><div class="frk">6</div><div class="fn">Z-score Prix 20j</div><div class="fb"><div class="ff" style="width:72%;background:var(--g)"></div></div><div class="fp">0.050</div></div>
      <div class="frow"><div class="frk">7</div><div class="fn">Q4 Saisonnalité</div><div class="fb"><div class="ff" style="width:67%;background:var(--g)"></div></div><div class="fp">0.046</div></div>
      <div class="frow"><div class="frk">8</div><div class="fn">Corr SP500/VIX 20j</div><div class="fb"><div class="ff" style="width:60%;background:var(--a)"></div></div><div class="fp">0.041</div></div>
      <div class="frow"><div class="frk">9</div><div class="fn">Delta VIX 1j</div><div class="fb"><div class="ff" style="width:56%;background:var(--a)"></div></div><div class="fp">0.039</div></div>
      <div class="frow"><div class="frk">10</div><div class="fn">Rendement 3j</div><div class="fb"><div class="ff" style="width:55%;background:var(--a)"></div></div><div class="fp">0.038</div></div>
    </div>
    <div id="shap-stress" style="display:none">
      <div class="frow"><div class="frk">1</div><div class="fn">Fear &amp; Greed Extrême</div><div class="fb"><div class="ff" style="width:100%;background:var(--r)"></div></div><div class="fp">0.075</div></div>
      <div class="frow"><div class="frk">2</div><div class="fn">ATR normalisé</div><div class="fb"><div class="ff" style="width:71%;background:var(--r)"></div></div><div class="fp">0.054</div></div>
      <div class="frow"><div class="frk">3</div><div class="fn">VIX Spike vs MA20</div><div class="fb"><div class="ff" style="width:66%;background:var(--r)"></div></div><div class="fp">0.050</div></div>
      <div class="frow"><div class="frk">4</div><div class="fn">Bollinger Z-score</div><div class="fb"><div class="ff" style="width:72%;background:var(--a)"></div></div><div class="fp">0.054</div></div>
      <div class="frow"><div class="frk">5</div><div class="fn">Drawdown 50j</div><div class="fb"><div class="ff" style="width:55%;background:var(--a)"></div></div><div class="fp">0.041</div></div>
      <div class="frow"><div class="frk">6</div><div class="fn">Delta VIX 1j</div><div class="fb"><div class="ff" style="width:55%;background:var(--a)"></div></div><div class="fp">0.041</div></div>
      <div class="frow"><div class="frk">7</div><div class="fn">VIX Niveau</div><div class="fb"><div class="ff" style="width:56%;background:var(--a)"></div></div><div class="fp">0.042</div></div>
      <div class="frow"><div class="frk">8</div><div class="fn">Drawdown 20j</div><div class="fb"><div class="ff" style="width:47%;background:var(--t3)"></div></div><div class="fp">0.035</div></div>
      <div class="frow"><div class="frk">9</div><div class="fn">Corrélation SP/VIX</div><div class="fb"><div class="ff" style="width:45%;background:var(--t3)"></div></div><div class="fp">0.034</div></div>
      <div class="frow"><div class="frk">10</div><div class="fn">RSI 2 survente</div><div class="fb"><div class="ff" style="width:45%;background:var(--t3)"></div></div><div class="fp">0.034</div></div>
    </div>
  </div>
</div>

<!-- FOOTER -->
<div style="border-top:1px solid var(--bd);padding-top:12px;display:flex;justify-content:space-between;align-items:center">
  <div style="font-size:10px;font-family:var(--mono);color:var(--t3)">RF 30% · XGBoost 50% · LR 20% · Walk-forward 5 folds · Zero data leakage</div>
  <div style="font-size:10px;font-family:var(--mono);color:var(--t3)">Auto-refresh 60s · <span id="cd">60</span>s</div>
</div>

</div><!-- .wrap -->

<script>
let charts = {};
let cdVal  = 60;
let cdTimer;

// Horloge
setInterval(()=>{ document.getElementById('clock').textContent = new Date().toLocaleTimeString('fr-FR'); }, 1000);
document.getElementById('clock').textContent = new Date().toLocaleTimeString('fr-FR');

// Countdown
function startCd() {
  cdVal = 60;
  clearInterval(cdTimer);
  cdTimer = setInterval(()=>{
    cdVal--;
    const el = document.getElementById('cd');
    if(el) el.textContent = cdVal;
    if(cdVal <= 0) load();
  }, 1000);
}

// Helpers
const $ = id => document.getElementById(id);
const fmt = (v,d=2) => parseFloat(v||0).toFixed(d);
const fmtP = v => parseFloat(v||0).toLocaleString('fr-FR',{minimumFractionDigits:2,maximumFractionDigits:2});
const sign = v => v>=0?'+':'';
const col  = v => v>=0?'var(--g)':'var(--r)';
const colClass = v => v>=0?'g':'r';

function legend(id, items) {
  const el = $(id);
  if(!el) return;
  el.innerHTML = items.map(i=>
    `<span style="display:flex;align-items:center;gap:5px;font-size:10px;font-family:var(--mono);color:var(--t2)">
      <span style="width:12px;height:2px;background:${i.c};display:inline-block;border-radius:1px"></span>${i.l}
    </span>`).join('');
}

function mkChart(id, labels, datasets) {
  const ctx = $(id);
  if(!ctx) return null;
  if(charts[id]) { charts[id].destroy(); }
  charts[id] = new Chart(ctx, {
    type:'line', data:{labels, datasets},
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip:{
          mode:'index', intersect:false,
          backgroundColor:'#161b22', borderColor:'#30363d', borderWidth:1,
          titleColor:'#8b949e', bodyColor:'#f0f6fc',
          callbacks:{label:c=>` ${c.dataset.label}: ${parseFloat(c.parsed.y).toFixed(1)}%`}
        }
      },
      scales:{
        x:{grid:{color:'rgba(240,246,252,.04)'}, ticks:{color:'#484f58',font:{size:9,family:'JetBrains Mono'},maxRotation:0,maxTicksLimit:7}},
        y:{grid:{color:'rgba(240,246,252,.04)'}, ticks:{color:'#484f58',font:{size:9,family:'JetBrains Mono'},callback:v=>v.toFixed(0)+'%'}}
      },
      interaction:{mode:'index',intersect:false}
    }
  });
  return charts[id];
}

function ds(label, data, color, dash) {
  return {
    label, data,
    borderColor:color, backgroundColor:color+'14',
    fill:!dash, tension:.4, pointRadius:0, borderWidth:dash?1.5:2,
    borderDash:dash?[5,3]:undefined
  };
}

// SHAP tabs
function shap(r, btn) {
  $('shap-calme').style.display  = r==='calme'  ? '':'none';
  $('shap-stress').style.display = r==='stress' ? '':'none';
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  btn.classList.add('on');
}

// MAIN LOAD
async function load() {
  startCd();
  try {
    const d = await fetch('/api/data').then(r=>r.json());
    $('ts').textContent = d.ts || '---';

    // KPI
    const sp = d.marche?.sp500 || {};
    const vx = d.marche?.vix  || {};
    const sig = d.signal || {};
    const ten = d.tendance || {};

    $('k-sp').textContent  = fmtP(sp.prix);
    $('k-sp').className    = 'sc-val '+(sp.var>=0?'g':'r');
    $('k-spv').innerHTML   = `<span class="${sp.var>=0?'g':'r'}">${sp.var>=0?'▲':'▼'} ${fmt(Math.abs(sp.var))}%</span>&nbsp;aujourd'hui`;

    $('k-vix').textContent = fmt(vx.prix,1);
    const regColors = {calme:'var(--b)',stress:'var(--a)',crash:'var(--r)'};
    const regLabels = {calme:'CALME',stress:'STRESS',crash:'CRASH'};
    $('k-reg').innerHTML   = `<span style="color:${regColors[sig.regime]||'var(--b)'};font-size:10px;font-family:var(--mono);background:${regColors[sig.regime]||'var(--b)'}18;padding:2px 8px;border-radius:4px">${regLabels[sig.regime]||'CALME'}</span>`;

    const pc = sig.proba||50;
    $('k-prob').textContent = pc + '%';
    $('k-prob').className   = 'sc-val '+(pc>=55?'g':pc<45?'r':'a');

    $('k-expo').textContent = (sig.expo||0) + '%';
    $('k-expo').className   = 'sc-val '+(sig.expo>=50?'g':sig.expo>0?'a':'r');
    $('k-sig').textContent  = sig.signal || '---';

    // Signal badge
    const sigEl = $('sig-tag');
    const cls   = {green:'sg',red:'sr',amber:'sa',gray:'sx'}[sig.couleur]||'sx';
    const ico   = {green:'▲',red:'▼',amber:'◆',gray:'→'}[sig.couleur]||'→';
    sigEl.textContent = ico+' '+(sig.signal||'---');
    sigEl.className   = 'sig-tag '+cls;
    $('sig-meta').textContent = `Conviction ${sig.conviction||'---'} · Score ${sig.score>=0?'+':''}${sig.score}/100 · Régime ${(sig.regime||'calme').toUpperCase()}`;
    $('sig-date').textContent = sig.date||'---';

    // Barre proba
    const pEl = $('pf');
    pEl.style.width      = Math.min(100,Math.max(0,pc))+'%';
    pEl.style.background = pc>=60?'var(--g)':pc<40?'var(--r)':'var(--a)';

    // Horizons
    const mlScore = (pc-50)*2;
    [['ct',ten.score_ct||0,'var(--g)'],
     ['mt',ten.score_mt||0,'var(--g)'],
     ['lt',ten.score_lt||0,'var(--g)'],
     ['ml',mlScore,'var(--b)']
    ].forEach(([id,sc,c])=>{
      const h = $('h-'+id), v = $('v-'+id);
      if(h){ h.style.width=Math.min(50,Math.abs(sc)/2)+'%'; h.style.background=sc>=0?c:'var(--r)'; h.style.left=sc>=0?'50%':'calc(50% - '+Math.min(50,Math.abs(sc)/2)+'%)'; }
      if(v){ v.textContent=(sc>=0?'+':'')+fmt(sc,0); v.className='hval '+(sc>=0?'g':'r'); }
    });

    // Niveaux
    ['ma20','ma50','ma200'].forEach(k=>{ const el=$(k); if(el) el.textContent=fmtP(ten[k]||0); });
    $('sup') && ($('sup').textContent  = fmtP(ten.support1||0));
    $('res') && ($('res').textContent  = fmtP(ten.resistance||0));
    $('rsi') && ($('rsi').textContent  = fmt(ten.rsi||0,1));

    // Indicateurs
    const setIv = (id,v,d=4)=>{ const e=$(id); if(e){e.textContent=v>=0?'+'+fmt(v,d):fmt(v,d); e.className='iv '+(v>=0?'g':'r');} };
    setIv('i-macd', ten.macd_hist||0, 4);
    setIv('i-bb',   ten.bb_z||0,     2);
    setIv('i-mom',  ten.mom5||0,     2);
    setIv('i-e20',  ten.ecart20||0,  2);
    setIv('i-e50',  ten.ecart50||0,  2);
    setIv('i-e200', ten.ecart200||0, 2);

    // Assets
    const assetEl = $('assets');
    if(assetEl){
      const keys = ['sp500','vix','bitcoin','gold','dxy'];
      assetEl.innerHTML = keys.map(k=>{
        const a = d.marche?.[k]||{};
        const vv = a.var||0;
        const badge = k==='vix'
          ? (vv<0?'<span class="abadge bag">BON</span>':'<span class="abadge bax">NEU</span>')
          : `<span class="abadge ${Math.abs(vv)>=1?(vv>0?'bag':'bar'):'bax'}">${Math.abs(vv)>=1?(vv>0?'ACHAT':'VENTE'):'NEU'}</span>`;
        return `<div class="arow">
          <div class="atic b">${a.ticker||k}</div>
          <div class="an">${a.label||k}</div>
          <div class="ap">${fmtP(a.prix)}</div>
          <div class="av ${vv>=0?'g':'r'}">${vv>=0?'▲':'▼'} ${fmt(Math.abs(vv))}%</div>
          ${badge}
        </div>`;
      }).join('');
    }

    // Charts perf
    const p = d.perf||{};
    ['calme','stress'].forEach(r=>{
      const pd = p[r]||{};
      const lbs = pd.dates||[];
      if(!lbs.length) return;
      legend('leg-'+r, [
        {c:'var(--g)',l:'ML Alloc.'},
        {c:'#484f58',l:'Buy&Hold'},
        {c:'var(--b)',l:'Binaire'},
      ]);
      mkChart('ch-'+r, lbs, [
        ds('ML Alloc.', pd.ml||[], 'var(--g)'),
        ds('Buy&Hold',  pd.bh||[], '#484f58', true),
        ds('Binaire',   pd.bin||[],'var(--b)', true),
      ]);
    });

    // Historique
    const hist = d.historique||[];
    const hbox = $('hist-box');
    const hcount = $('hist-count');
    if(hcount) hcount.textContent = hist.length+' entrées';
    if(hbox && hist.length>0){
      const rows = [...hist].reverse().map(row=>{
        const sc = parseFloat(row.score_global||0);
        const c  = sc>20?'var(--g)':sc<-20?'var(--r)':'var(--a)';
        return `<tr>
          <td style="color:var(--t2)">${(row.date||'').substring(0,10)}</td>
          <td style="color:${c};font-weight:600">${sc>=0?'+':''}${sc.toFixed(1)}</td>
          <td style="color:${parseFloat(row.score_ct||0)>=0?'var(--g)':'var(--r)'}">${parseFloat(row.score_ct||0).toFixed(0)}</td>
          <td style="color:${parseFloat(row.score_mt||0)>=0?'var(--g)':'var(--r)'}">${parseFloat(row.score_mt||0).toFixed(0)}</td>
          <td style="color:${parseFloat(row.score_lt||0)>=0?'var(--g)':'var(--r)'}">${parseFloat(row.score_lt||0).toFixed(0)}</td>
          <td>${(row.tendance_global||'---').replace(/[▲▽▼△↑↓]/g,'').trim()}</td>
          <td style="color:${row.conviction==='FORTE'?'var(--g)':row.conviction==='FAIBLE'?'var(--r)':'var(--a)'}">${row.conviction||'---'}</td>
          <td>${parseFloat(row.vix||0).toFixed(1)}</td>
          <td style="text-transform:capitalize">${row.regime||'---'}</td>
        </tr>`;
      }).join('');
      hbox.innerHTML = `<table class="htable">
        <thead><tr><th>Date</th><th>Score</th><th>CT</th><th>MT</th><th>LT</th><th>Tendance</th><th>Conv.</th><th>VIX</th><th>Régime</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
    } else if(hbox){
      hbox.innerHTML = '<div class="muted" style="font-size:11px;font-family:var(--mono)">Aucun historique — lance 4_prediction_tendance.py d\'abord</div>';
    }

  } catch(e) {
    console.error('Erreur load:', e);
    $('ts').textContent = 'Erreur: '+e.message;
  }
}

load();
setInterval(load, 60000);
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(PAGE)

if __name__ == '__main__':
    print("="*55)
    print("  S&P 500 ML PREDICTOR — DASHBOARD v2")
    print("="*55)
    print(f"  SQLite  : {DB_PATH}")
    print(f"  URL     : http://localhost:5000")
    print(f"  Refresh : toutes les 60 secondes")
    print("="*55)
    if not os.path.exists(DB_PATH):
        print(f"\n  ATTENTION : {DB_PATH} introuvable")
        print("  Lance : python 01_init_data.py")
    app.run(host='0.0.0.0', port=5000, debug=False)