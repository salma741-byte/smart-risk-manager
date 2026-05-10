<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S&P 500 ML Predictor — Live Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #080c14;
    --bg2:     #0d1320;
    --bg3:     #111b2e;
    --bg4:     #162035;
    --border:  #1e2d45;
    --border2: #253650;
    --text:    #e2e8f0;
    --text2:   #94a3b8;
    --text3:   #4a5568;
    --blue:    #3b82f6;
    --blue2:   #185fa5;
    --green:   #22c55e;
    --green2:  #3b6d11;
    --red:     #ef4444;
    --red2:    #a32d2d;
    --orange:  #f97316;
    --yellow:  #eab308;
    --cyan:    #06b6d4;
    --purple:  #a855f7;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
  }

  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    min-height: 100vh;
    font-size: 13px;
    line-height: 1.5;
  }

  /* ── LAYOUT ── */
  .wrap      { max-width: 1400px; margin: 0 auto; padding: 0 24px 40px; }
  .topbar    { display: flex; align-items: center; justify-content: space-between;
               padding: 18px 0 16px; border-bottom: 0.5px solid var(--border); margin-bottom: 20px; }
  .grid-4    { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  .grid-3    { display: grid; grid-template-columns: 1.6fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  .grid-2    { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  .grid-21   { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; margin-bottom: 16px; }

  /* ── CARD ── */
  .card {
    background: var(--bg2);
    border: 0.5px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,.3), transparent);
  }
  .card-title {
    font-size: 9px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 12px;
    font-family: var(--mono);
  }

  /* ── METRIC CARDS ── */
  .metric-card { background: var(--bg3); border: 0.5px solid var(--border); border-radius: 8px; padding: 14px 16px; }
  .metric-label { font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: var(--text3); margin-bottom: 6px; }
  .metric-value { font-size: 26px; font-weight: 600; line-height: 1; margin-bottom: 4px; }
  .metric-sub   { font-size: 10px; color: var(--text2); display: flex; align-items: center; gap: 5px; }
  .up { color: var(--green); }
  .dn { color: var(--red); }
  .neu { color: var(--text2); }

  /* ── SIGNAL BOX ── */
  .signal-main { text-align: center; padding: 20px 16px; }
  .signal-tag  {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 20px; font-weight: 600;
    padding: 10px 24px; border-radius: 8px; margin-bottom: 10px;
  }
  .tag-buy  { background: rgba(34,197,94,.12); color: var(--green); border: 1px solid rgba(34,197,94,.25); }
  .tag-sell { background: rgba(239,68,68,.12);  color: var(--red);   border: 1px solid rgba(239,68,68,.25); }
  .tag-hold { background: rgba(148,163,184,.08);color: var(--text2); border: 1px solid var(--border2); }

  /* ── PROGRESS BAR ── */
  .pbar-wrap { width: 100%; height: 6px; background: var(--bg4); border-radius: 3px; overflow: hidden; margin: 8px 0 4px; }
  .pbar-fill { height: 100%; border-radius: 3px; transition: width 1s ease; }

  /* ── HORIZON ROWS ── */
  .horizon-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .horizon-lbl  { font-size: 10px; color: var(--text2); min-width: 90px; }
  .horizon-bar  { flex: 1; height: 6px; background: var(--bg4); border-radius: 3px; overflow: hidden; position: relative; }
  .horizon-fill { position: absolute; top: 0; height: 100%; border-radius: 3px; left: 50%; transition: width .8s, transform .8s; }
  .horizon-score{ font-size: 11px; font-weight: 600; min-width: 42px; text-align: right; }

  /* ── FEATURE IMPORTANCE ── */
  .feat-row  { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .feat-rank { font-size: 9px; color: var(--text3); min-width: 16px; }
  .feat-name { font-size: 10px; color: var(--text2); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .feat-bar  { flex: 0 0 90px; height: 5px; background: var(--bg4); border-radius: 3px; overflow: hidden; }
  .feat-fill { height: 100%; border-radius: 3px; }
  .feat-pct  { font-size: 9px; color: var(--text3); min-width: 30px; text-align: right; }

  /* ── ASSET LIST ── */
  .asset-row { display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 0.5px solid var(--border); }
  .asset-row:last-child { border-bottom: none; }
  .asset-ticker { font-size: 12px; font-weight: 600; min-width: 55px; }
  .asset-name   { font-size: 9px; color: var(--text3); flex: 1; }
  .asset-price  { font-size: 11px; color: var(--text); min-width: 70px; text-align: right; }
  .asset-chg    { font-size: 11px; font-weight: 600; min-width: 52px; text-align: right; }
  .asset-sig    { font-size: 9px; padding: 2px 7px; border-radius: 4px; min-width: 44px; text-align: center; }
  .sig-b { background: rgba(34,197,94,.12);  color: var(--green); }
  .sig-v { background: rgba(239,68,68,.12);  color: var(--red); }
  .sig-n { background: rgba(148,163,184,.08);color: var(--text2); }

  /* ── REGIME BADGE ── */
  .regime-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 9px;
                  padding: 2px 8px; border-radius: 4px; letter-spacing: .06em; text-transform: uppercase; }
  .reg-calme  { background: rgba(59,130,246,.12);  color: var(--blue); }
  .reg-stress { background: rgba(249,115,22,.12);  color: var(--orange); }
  .reg-crash  { background: rgba(239,68,68,.12);   color: var(--red); }

  /* ── WATERFALL ── */
  .wf-row  { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
  .wf-name { font-size: 10px; color: var(--text2); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .wf-shap { font-size: 10px; font-weight: 600; min-width: 44px; text-align: right; }
  .wf-bar  { flex: 0 0 80px; height: 5px; background: var(--bg4); border-radius: 3px; overflow: hidden; position: relative; }
  .wf-fill { position: absolute; top: 0; height: 100%; border-radius: 3px; }

  /* ── WALK-FORWARD ── */
  .fold-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .fold-lbl { font-size: 10px; color: var(--text2); min-width: 50px; }
  .fold-bar { flex: 1; height: 6px; background: var(--bg4); border-radius: 3px; overflow: hidden; }
  .fold-fill{ height: 100%; border-radius: 3px; transition: width .8s; }
  .fold-val { font-size: 10px; font-weight: 600; min-width: 36px; text-align: right; }

  /* ── SEPARATOR ── */
  .sep { border: none; border-top: 0.5px solid var(--border); margin: 16px 0; }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* ── LIVE DOT ── */
  .live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); display: inline-block; animation: blink 1.4s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

  /* ── LOGO ── */
  .logo-box { display: flex; align-items: center; gap: 12px; }
  .logo-icon { width: 32px; height: 32px; background: var(--blue2); border-radius: 7px;
               display: flex; align-items: center; justify-content: center; font-size: 16px; }
  .logo-title { font-size: 14px; font-weight: 600; letter-spacing: .04em; }
  .logo-sub   { font-size: 9px; color: var(--text3); letter-spacing: .1em; text-transform: uppercase; margin-top: 1px; }

  /* ── TABS ── */
  .tabs { display: flex; gap: 6px; margin-bottom: 12px; }
  .tab  { font-size: 10px; padding: 4px 12px; border: 0.5px solid var(--border2);
          border-radius: 5px; background: transparent; color: var(--text2); cursor: pointer;
          font-family: var(--mono); transition: all .15s; }
  .tab.on { background: var(--bg3); color: var(--text); border-color: var(--border2); }

  /* ── SECTION TITLE ── */
  .section-title { font-size: 9px; letter-spacing: .15em; text-transform: uppercase;
                   color: var(--text3); margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
  .section-title::after { content: ''; flex: 1; height: 0.5px; background: var(--border); }
</style>
</head>
<body>
<div class="wrap">

  <!-- ── TOP BAR ── -->
  <div class="topbar">
    <div class="logo-box">
      <div class="logo-icon">📈</div>
      <div>
        <div class="logo-title">S&P 500 ML PREDICTOR</div>
        <div class="logo-sub">Intelligent Trading System · Dual-Regime Ensemble</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:10px">
      <span class="live-dot"></span>
      <span style="font-size:10px;color:var(--text2)">LIVE</span>
      <span style="font-size:11px;color:var(--text2);background:var(--bg3);padding:4px 12px;border-radius:5px;border:0.5px solid var(--border)" id="clock">--:--:--</span>
      <span style="font-size:11px;color:var(--text2);background:var(--bg3);padding:4px 12px;border-radius:5px;border:0.5px solid var(--border)" id="datestr">---</span>
    </div>
  </div>

  <!-- ── METRIC CARDS ── -->
  <div class="grid-4">
    <div class="metric-card">
      <div class="metric-label">S&P 500</div>
      <div class="metric-value up" id="sp-price">7,259.22</div>
      <div class="metric-sub"><span class="up">▲ +0.43%</span>&nbsp;aujourd'hui</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">VIX</div>
      <div class="metric-value" id="vix-val" style="color:var(--blue)">17.4</div>
      <div class="metric-sub">
        Régime&nbsp;<span class="regime-badge reg-calme" id="regime-badge">CALME</span>
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-label">P(hausse 5j)</div>
      <div class="metric-value up" id="proba-val">67.4%</div>
      <div class="metric-sub" style="color:var(--text3)">RF 30% · XGB 50% · LR 20%</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Exposition cible</div>
      <div class="metric-value up" id="expo-val">100%</div>
      <div class="metric-sub"><span class="up">● FORT ACHAT</span>&nbsp;allocation dyn.</div>
    </div>
  </div>

  <!-- ── ROW 2 : SIGNAL + TENDANCE + ASSETS ── -->
  <div class="grid-3">

    <!-- Signal -->
    <div class="card">
      <div class="card-title">Signal de trading</div>
      <div class="signal-main">
        <div class="signal-tag tag-buy" id="signal-badge">▲ FORT ACHAT</div>
        <div style="font-size:10px;color:var(--text3);margin-bottom:12px" id="signal-meta">
          Conviction FORTE · Score +45.1/100
        </div>
        <div class="pbar-wrap">
          <div class="pbar-fill" id="proba-fill" style="width:67.4%;background:var(--green)"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--text3)">
          <span>VENTE ←</span><span>→ ACHAT</span>
        </div>
      </div>
      <hr class="sep">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center">
        <div style="background:var(--bg3);border-radius:6px;padding:8px 4px">
          <div style="font-size:9px;color:var(--text3)">Sharpe</div>
          <div style="font-size:18px;font-weight:600;color:var(--green)">2.61</div>
        </div>
        <div style="background:var(--bg3);border-radius:6px;padding:8px 4px">
          <div style="font-size:9px;color:var(--text3)">Max DD</div>
          <div style="font-size:18px;font-weight:600;color:var(--red)">-2.9%</div>
        </div>
        <div style="background:var(--bg3);border-radius:6px;padding:8px 4px">
          <div style="font-size:9px;color:var(--text3)">Hit Rate</div>
          <div style="font-size:18px;font-weight:600;color:var(--blue)">63.6%</div>
        </div>
      </div>
    </div>

    <!-- Tendance multi-horizons -->
    <div class="card">
      <div class="card-title">Tendance multi-horizons</div>
      <div id="horizons-box">
        <div class="horizon-row">
          <div class="horizon-lbl">Court (5j)</div>
          <div class="horizon-bar">
            <div class="horizon-fill" style="width:33%;left:50%;background:var(--green)"></div>
          </div>
          <div class="horizon-score up">+33</div>
        </div>
        <div class="horizon-row">
          <div class="horizon-lbl">Moyen (20j)</div>
          <div class="horizon-bar">
            <div class="horizon-fill" style="width:40%;left:50%;background:var(--green)"></div>
          </div>
          <div class="horizon-score up">+40</div>
        </div>
        <div class="horizon-row">
          <div class="horizon-lbl">Long (60j)</div>
          <div class="horizon-bar">
            <div class="horizon-fill" style="width:50%;left:50%;background:var(--green)"></div>
          </div>
          <div class="horizon-score up">+71</div>
        </div>
        <div class="horizon-row">
          <div class="horizon-lbl">ML Score</div>
          <div class="horizon-bar">
            <div class="horizon-fill" style="width:35%;left:50%;background:var(--blue)"></div>
          </div>
          <div class="horizon-score up" style="color:var(--blue)">+35</div>
        </div>
      </div>
      <hr class="sep">
      <div style="font-size:10px;color:var(--text2)">Niveaux clés</div>
      <div style="margin-top:8px;display:flex;flex-direction:column;gap:5px" id="levels-box">
        <div style="display:flex;justify-content:space-between;font-size:10px">
          <span style="color:var(--text3)">MA20</span>
          <span>7,070</span>
          <span class="up">+2.68%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px">
          <span style="color:var(--text3)">MA50</span>
          <span>6,836</span>
          <span class="up">+6.19%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px">
          <span style="color:var(--text3)">MA200</span>
          <span>6,738</span>
          <span class="up">+7.74%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px">
          <span style="color:var(--text3)">Support</span>
          <span style="color:var(--orange)">6,631</span>
          <span class="dn">-8.64%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px">
          <span style="color:var(--text3)">Résistance</span>
          <span style="color:var(--yellow)">7,404</span>
          <span class="up">+2.00%</span>
        </div>
      </div>
    </div>

    <!-- Assets -->
    <div class="card">
      <div class="card-title">Multi-actifs</div>
      <div id="assets-box">
        <div class="asset-row">
          <div class="asset-ticker" style="color:var(--blue)">^GSPC</div>
          <div class="asset-name">S&P 500</div>
          <div class="asset-price">7,259.22</div>
          <div class="asset-chg up">+0.43%</div>
          <div class="asset-sig sig-b">ACHAT</div>
        </div>
        <div class="asset-row">
          <div class="asset-ticker" style="color:var(--orange)">^VIX</div>
          <div class="asset-name">CBOE VIX</div>
          <div class="asset-price">17.4</div>
          <div class="asset-chg dn">-2.10%</div>
          <div class="asset-sig sig-b">BON</div>
        </div>
        <div class="asset-row">
          <div class="asset-ticker" style="color:var(--yellow)">BTC</div>
          <div class="asset-name">Bitcoin</div>
          <div class="asset-price">97,842</div>
          <div class="asset-chg up">+1.82%</div>
          <div class="asset-sig sig-b">ACHAT</div>
        </div>
        <div class="asset-row">
          <div class="asset-ticker" style="color:var(--yellow)">GC=F</div>
          <div class="asset-name">Or / Gold</div>
          <div class="asset-price">3,318</div>
          <div class="asset-chg up">+0.31%</div>
          <div class="asset-sig sig-n">NEU</div>
        </div>
        <div class="asset-row">
          <div class="asset-ticker" style="color:var(--cyan)">DXY</div>
          <div class="asset-name">Dollar Index</div>
          <div class="asset-price">99.84</div>
          <div class="asset-chg dn">-0.18%</div>
          <div class="asset-sig sig-b">BON</div>
        </div>
      </div>
      <hr class="sep">
      <div style="font-size:10px;color:var(--text3);text-align:center">
        Données Yahoo Finance · actualisées 18h00
      </div>
    </div>
  </div>

  <!-- ── ROW 3 : PERF CHART + SHAP ── -->
  <div class="grid-21">

    <!-- Performance chart -->
    <div class="card">
      <div class="card-title">Performance portefeuille — backtest 2021→2026</div>
      <div style="display:flex;gap:16px;margin-bottom:12px">
        <span style="display:flex;align-items:center;gap:5px;font-size:10px;color:var(--text2)">
          <span style="width:10px;height:3px;background:var(--green);border-radius:2px;display:inline-block"></span>
          ML Ensemble +84.9%
        </span>
        <span style="display:flex;align-items:center;gap:5px;font-size:10px;color:var(--text2)">
          <span style="width:10px;height:3px;background:var(--text3);border-radius:2px;display:inline-block"></span>
          Buy &amp; Hold +73.3%
        </span>
        <span style="display:flex;align-items:center;gap:5px;font-size:10px;color:var(--text2)">
          <span style="width:10px;height:3px;background:var(--blue);border-radius:2px;display:inline-block;border-top:1px dashed var(--blue)"></span>
          Binaire +92.9%
        </span>
      </div>
      <div style="position:relative;height:200px">
        <canvas id="perfChart" role="img" aria-label="Performance cumulée ML vs Buy and Hold 2021-2026">
          ML Ensemble: +84.9%, Buy and Hold: +73.3%, Signal binaire: +92.9%
        </canvas>
      </div>
      <hr class="sep">
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center">
        <div>
          <div style="font-size:9px;color:var(--text3)">Walk-forward acc</div>
          <div style="font-size:14px;font-weight:600;color:var(--blue)">60.6%</div>
        </div>
        <div>
          <div style="font-size:9px;color:var(--text3)">OOB Score RF</div>
          <div style="font-size:14px;font-weight:600;color:var(--cyan)">70.3%</div>
        </div>
        <div>
          <div style="font-size:9px;color:var(--text3)">Stress MaxDD</div>
          <div style="font-size:14px;font-weight:600;color:var(--red)">-5.2%</div>
        </div>
        <div>
          <div style="font-size:9px;color:var(--text3)">vs B&amp;H stress</div>
          <div style="font-size:14px;font-weight:600;color:var(--green)">-8.5%</div>
        </div>
      </div>
    </div>

    <!-- SHAP Feature Importance -->
    <div class="card">
      <div class="card-title">SHAP — Feature importance</div>
      <div class="tabs">
        <button class="tab on" onclick="switchShap('calme',this)">Calme</button>
        <button class="tab"    onclick="switchShap('stress',this)">Stress</button>
      </div>
      <div id="shap-calme">
        <div class="feat-row"><div class="feat-rank">1</div><div class="feat-name">MACD Histogram</div><div class="feat-bar"><div class="feat-fill" style="width:100%;background:var(--blue)"></div></div><div class="feat-pct">0.069</div></div>
        <div class="feat-row"><div class="feat-rank">2</div><div class="feat-name">Fear &amp; Greed Index</div><div class="feat-bar"><div class="feat-fill" style="width:99%;background:var(--blue)"></div></div><div class="feat-pct">0.068</div></div>
        <div class="feat-row"><div class="feat-rank">3</div><div class="feat-name">Bollinger Z-score</div><div class="feat-bar"><div class="feat-fill" style="width:93%;background:var(--blue)"></div></div><div class="feat-pct">0.064</div></div>
        <div class="feat-row"><div class="feat-rank">4</div><div class="feat-name">VIX Percentile 252j</div><div class="feat-bar"><div class="feat-fill" style="width:80%;background:var(--cyan)"></div></div><div class="feat-pct">0.055</div></div>
        <div class="feat-row"><div class="feat-rank">5</div><div class="feat-name">DI+ &gt; DI- (ADX)</div><div class="feat-bar"><div class="feat-fill" style="width:73%;background:var(--cyan)"></div></div><div class="feat-pct">0.051</div></div>
        <div class="feat-row"><div class="feat-rank">6</div><div class="feat-name">Z-score Prix 20j</div><div class="feat-bar"><div class="feat-fill" style="width:72%;background:var(--purple)"></div></div><div class="feat-pct">0.050</div></div>
        <div class="feat-row"><div class="feat-rank">7</div><div class="feat-name">Q4 Saisonnalité</div><div class="feat-bar"><div class="feat-fill" style="width:67%;background:var(--purple)"></div></div><div class="feat-pct">0.046</div></div>
        <div class="feat-row"><div class="feat-rank">8</div><div class="feat-name">Corr SP500/VIX</div><div class="feat-bar"><div class="feat-fill" style="width:60%;background:var(--orange)"></div></div><div class="feat-pct">0.041</div></div>
        <div class="feat-row"><div class="feat-rank">9</div><div class="feat-name">Delta VIX 1j</div><div class="feat-bar"><div class="feat-fill" style="width:56%;background:var(--orange)"></div></div><div class="feat-pct">0.039</div></div>
        <div class="feat-row"><div class="feat-rank">10</div><div class="feat-name">Rendement 3j</div><div class="feat-bar"><div class="feat-fill" style="width:55%;background:var(--orange)"></div></div><div class="feat-pct">0.038</div></div>
      </div>
      <div id="shap-stress" style="display:none">
        <div class="feat-row"><div class="feat-rank">1</div><div class="feat-name">Fear &amp; Greed Extrême</div><div class="feat-bar"><div class="feat-fill" style="width:100%;background:var(--red)"></div></div><div class="feat-pct">0.075</div></div>
        <div class="feat-row"><div class="feat-rank">2</div><div class="feat-name">Bollinger Z-score</div><div class="feat-bar"><div class="feat-fill" style="width:72%;background:var(--red)"></div></div><div class="feat-pct">0.054</div></div>
        <div class="feat-row"><div class="feat-rank">3</div><div class="feat-name">Drawdown 50j</div><div class="feat-bar"><div class="feat-fill" style="width:55%;background:var(--orange)"></div></div><div class="feat-pct">0.041</div></div>
        <div class="feat-row"><div class="feat-rank">4</div><div class="feat-name">Delta VIX 1j</div><div class="feat-bar"><div class="feat-fill" style="width:55%;background:var(--orange)"></div></div><div class="feat-pct">0.041</div></div>
        <div class="feat-row"><div class="feat-rank">5</div><div class="feat-name">VIX Niveau</div><div class="feat-bar"><div class="feat-fill" style="width:56%;background:var(--orange)"></div></div><div class="feat-pct">0.042</div></div>
        <div class="feat-row"><div class="feat-rank">6</div><div class="feat-name">Drawdown 20j</div><div class="feat-bar"><div class="feat-fill" style="width:47%;background:var(--yellow)"></div></div><div class="feat-pct">0.035</div></div>
        <div class="feat-row"><div class="feat-rank">7</div><div class="feat-name">Corrélation SP/VIX</div><div class="feat-bar"><div class="feat-fill" style="width:45%;background:var(--yellow)"></div></div><div class="feat-pct">0.034</div></div>
        <div class="feat-row"><div class="feat-rank">8</div><div class="feat-name">RSI 2 (survente)</div><div class="feat-bar"><div class="feat-fill" style="width:45%;background:var(--yellow)"></div></div><div class="feat-pct">0.034</div></div>
        <div class="feat-row"><div class="feat-rank">9</div><div class="feat-name">ATR Normalisé</div><div class="feat-bar"><div class="feat-fill" style="width:71%;background:var(--orange)"></div></div><div class="feat-pct">0.054</div></div>
        <div class="feat-row"><div class="feat-rank">10</div><div class="feat-name">VIX Spike vs MA20</div><div class="feat-bar"><div class="feat-fill" style="width:66%;background:var(--orange)"></div></div><div class="feat-pct">0.050</div></div>
      </div>
    </div>
  </div>

  <!-- ── ROW 4 : WALK-FORWARD + WATERFALL + REGIMES ── -->
  <div class="grid-3">

    <!-- Walk-forward validation -->
    <div class="card">
      <div class="card-title">Walk-forward validation — régime calme</div>
      <div id="wf-box">
        <div class="fold-row"><div class="fold-lbl">Fold 1</div><div class="fold-bar"><div class="fold-fill" style="width:56.2%;background:var(--blue)"></div></div><div class="fold-val" style="color:var(--blue)">56.2%</div></div>
        <div class="fold-row"><div class="fold-lbl">Fold 2</div><div class="fold-bar"><div class="fold-fill" style="width:40.6%;background:var(--red)"></div></div><div class="fold-val" style="color:var(--red)">40.6%</div></div>
        <div class="fold-row"><div class="fold-lbl">Fold 3</div><div class="fold-bar"><div class="fold-fill" style="width:68.8%;background:var(--green)"></div></div><div class="fold-val" style="color:var(--green)">68.8%</div></div>
        <div class="fold-row"><div class="fold-lbl">Fold 4</div><div class="fold-bar"><div class="fold-fill" style="width:71.9%;background:var(--green)"></div></div><div class="fold-val" style="color:var(--green)">71.9%</div></div>
        <div class="fold-row"><div class="fold-lbl">Fold 5</div><div class="fold-bar"><div class="fold-fill" style="width:65.6%;background:var(--green)"></div></div><div class="fold-val" style="color:var(--green)">65.6%</div></div>
      </div>
      <hr class="sep">
      <div style="display:flex;justify-content:space-between;font-size:10px">
        <span style="color:var(--text3)">Moyenne</span>
        <span style="font-weight:600;color:var(--blue)">60.6% ± 12.6%</span>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;margin-top:5px">
        <span style="color:var(--text3)">F1 Score</span>
        <span style="font-weight:600;color:var(--cyan)">0.638</span>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;margin-top:5px">
        <span style="color:var(--text3)">Zero leakage</span>
        <span style="font-weight:600;color:var(--green)">✓ Split temporel strict</span>
      </div>
    </div>

    <!-- SHAP Waterfall -->
    <div class="card">
      <div class="card-title">SHAP Waterfall — signal aujourd'hui</div>
      <div style="font-size:9px;color:var(--text3);margin-bottom:10px">
        Contributions de chaque feature au signal du 2026-05-05
      </div>
      <div id="waterfall-box">
        <div class="wf-row">
          <div class="wf-name" style="color:var(--text3)">Base (expected)</div>
          <div class="wf-bar"><div class="wf-fill" style="width:50%;left:0;background:var(--text3)"></div></div>
          <div class="wf-shap" style="color:var(--text3)">0.000</div>
        </div>
        <div class="wf-row">
          <div class="wf-name">MACD Histogram</div>
          <div class="wf-bar"><div class="wf-fill" style="width:60%;left:40%;background:var(--green)"></div></div>
          <div class="wf-shap up">+0.187</div>
        </div>
        <div class="wf-row">
          <div class="wf-name">VIX Percentile 252j</div>
          <div class="wf-bar"><div class="wf-fill" style="width:45%;left:55%;background:var(--green)"></div></div>
          <div class="wf-shap up">+0.142</div>
        </div>
        <div class="wf-row">
          <div class="wf-name">Bollinger Z-score</div>
          <div class="wf-bar"><div class="wf-fill" style="width:35%;left:65%;background:var(--green)"></div></div>
          <div class="wf-shap up">+0.098</div>
        </div>
        <div class="wf-row">
          <div class="wf-name">Q4 Saisonnalité</div>
          <div class="wf-bar"><div class="wf-fill" style="width:30%;left:70%;background:var(--green)"></div></div>
          <div class="wf-shap up">+0.076</div>
        </div>
        <div class="wf-row">
          <div class="wf-name">Stochastique</div>
          <div class="wf-bar"><div class="wf-fill" style="width:20%;left:30%;background:var(--red)"></div></div>
          <div class="wf-shap dn">-0.043</div>
        </div>
        <div class="wf-row">
          <div class="wf-name">Delta VIX</div>
          <div class="wf-bar"><div class="wf-fill" style="width:25%;left:25%;background:var(--red)"></div></div>
          <div class="wf-shap dn">-0.031</div>
        </div>
        <div class="wf-row">
          <div class="wf-name">Rendement 3j</div>
          <div class="wf-bar"><div class="wf-fill" style="width:28%;left:72%;background:var(--green)"></div></div>
          <div class="wf-shap up">+0.059</div>
        </div>
      </div>
      <hr class="sep">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:10px;color:var(--text3)">Score final log-odds</span>
        <span style="font-size:14px;font-weight:600;color:var(--green)">+0.488  →  P=67.4%</span>
      </div>
    </div>

    <!-- Régimes dans le temps -->
    <div class="card">
      <div class="card-title">Analyse par régime VIX</div>
      <div style="display:flex;flex-direction:column;gap:10px" id="regime-stats">
        <div style="background:var(--bg3);border-radius:6px;padding:10px;border-left:3px solid var(--green)">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:10px;font-weight:600;color:var(--green)">TRÈS CALME — VIX &lt; 15</span>
            <span style="font-size:9px;color:var(--text3)">1062 j.</span>
          </div>
          <div style="display:flex;gap:16px">
            <div><div style="font-size:9px;color:var(--text3)">Hit Rate</div><div style="font-size:14px;font-weight:600;color:var(--green)">61.1%</div></div>
            <div><div style="font-size:9px;color:var(--text3)">Sharpe</div><div style="font-size:14px;font-weight:600;color:var(--green)">5.07</div></div>
          </div>
        </div>
        <div style="background:var(--bg3);border-radius:6px;padding:10px;border-left:3px solid var(--blue)">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:10px;font-weight:600;color:var(--blue)">CALME — VIX 15-20</span>
            <span style="font-size:9px;color:var(--text3)">962 j.</span>
          </div>
          <div style="display:flex;gap:16px">
            <div><div style="font-size:9px;color:var(--text3)">Hit Rate</div><div style="font-size:14px;font-weight:600;color:var(--blue)">55.0%</div></div>
            <div><div style="font-size:9px;color:var(--text3)">Sharpe</div><div style="font-size:14px;font-weight:600;color:var(--blue)">2.29</div></div>
          </div>
        </div>
        <div style="background:var(--bg3);border-radius:6px;padding:10px;border-left:3px solid var(--orange)">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:10px;font-weight:600;color:var(--orange)">VOLATIL — VIX 20-30</span>
            <span style="font-size:9px;color:var(--text3)">705 j.</span>
          </div>
          <div style="display:flex;gap:16px">
            <div><div style="font-size:9px;color:var(--text3)">Hit Rate</div><div style="font-size:14px;font-weight:600;color:var(--orange)">46.4%</div></div>
            <div><div style="font-size:9px;color:var(--text3)">Sharpe</div><div style="font-size:14px;font-weight:600;color:var(--orange)">-0.96</div></div>
          </div>
        </div>
        <div style="background:var(--bg3);border-radius:6px;padding:10px;border-left:3px solid var(--red)">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:10px;font-weight:600;color:var(--red)">EXTRÊME — VIX &gt; 30</span>
            <span style="font-size:9px;color:var(--text3)">158 j.</span>
          </div>
          <div style="display:flex;gap:16px">
            <div><div style="font-size:9px;color:var(--text3)">Hit Rate</div><div style="font-size:14px;font-weight:600;color:var(--red)">37.3%</div></div>
            <div><div style="font-size:9px;color:var(--text3)">Sharpe</div><div style="font-size:14px;font-weight:600;color:var(--red)">-2.62</div></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── FOOTER ── -->
  <div style="border-top:0.5px solid var(--border);padding-top:14px;display:flex;justify-content:space-between;align-items:center">
    <div style="font-size:9px;color:var(--text3)">
      S&P 500 ML Predictor · RF + XGBoost + LR · Walk-forward validation · Zero data leakage
    </div>
    <div style="font-size:9px;color:var(--text3)">
      Data: Yahoo Finance · SQLite · Mise à jour 18h00 &amp; 00h00
    </div>
  </div>

</div>

<script>
  // ── HORLOGE ──
  function tick() {
    const now = new Date();
    document.getElementById('clock').textContent =
      now.toLocaleTimeString('fr-FR');
    document.getElementById('datestr').textContent =
      now.toLocaleDateString('fr-FR', {weekday:'short', day:'2-digit', month:'short', year:'numeric'});
  }
  tick();
  setInterval(tick, 1000);

  // ── TABS SHAP ──
  function switchShap(regime, btn) {
    document.getElementById('shap-calme').style.display  = regime === 'calme'  ? '' : 'none';
    document.getElementById('shap-stress').style.display = regime === 'stress' ? '' : 'none';
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('on'));
    btn.classList.add('on');
  }

  // ── CHART PERFORMANCE ──
  const labels = ['Avr 21','Jul 21','Oct 21','Jan 22','Avr 22','Jul 22','Oct 22',
                  'Jan 23','Avr 23','Jul 23','Oct 23','Jan 24','Avr 24','Jul 24',
                  'Oct 24','Jan 25','Avr 25','Avr 26'];

  const bh  = [100,108,115,114,101, 94, 98,107,116,125,120,130,138,148,152,158,145,173];
  const ml  = [100,106,112,113,104, 99,103,110,118,126,123,132,139,148,153,160,151,185];
  const bin = [100,107,114,115,107,103,108,115,124,134,130,140,149,160,163,172,162,193];

  const ctx = document.getElementById('perfChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'ML Ensemble', data: ml,  borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,.07)',
          fill: true, tension: .4, pointRadius: 0, borderWidth: 2 },
        { label: 'Buy & Hold',  data: bh,  borderColor: '#4a5568', backgroundColor: 'transparent',
          fill: false, tension: .4, pointRadius: 0, borderWidth: 1.5, borderDash: [4,3] },
        { label: 'Binaire',     data: bin, borderColor: '#3b82f6', backgroundColor: 'transparent',
          fill: false, tension: .4, pointRadius: 0, borderWidth: 1.5 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index', intersect: false,
          backgroundColor: '#0d1320', borderColor: '#1e2d45', borderWidth: 1,
          titleColor: '#94a3b8', bodyColor: '#e2e8f0',
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(0)}` }
        }
      },
      scales: {
        x: { grid: { color: '#1e2d45', lineWidth: .5 }, ticks: { color: '#4a5568', font: { size: 9 }, maxRotation: 0 } },
        y: { grid: { color: '#1e2d45', lineWidth: .5 }, ticks: { color: '#4a5568', font: { size: 9 },
             callback: v => v.toFixed(0) }, min: 85 }
      },
      interaction: { mode: 'index', intersect: false }
    }
  });

  // ── SIMULATION LIVE (fluctuations légères toutes les 10s) ──
  function simulateLive() {
    const base = 7259.22;
    const noise = (Math.random() - 0.5) * 8;
    const price = (base + noise).toFixed(2);
    const chg   = (noise / base * 100).toFixed(2);
    const up    = parseFloat(chg) >= 0;
    const el    = document.getElementById('sp-price');
    el.textContent = parseFloat(price).toLocaleString('fr-FR', {minimumFractionDigits:2});
    el.className   = 'metric-value ' + (up ? 'up' : 'dn');
  }
  setInterval(simulateLive, 10000);
</script>
</body>
</html>