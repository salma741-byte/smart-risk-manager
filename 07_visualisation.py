# ============================================================
#  2_train_extreme.py — MODÈLE MOUVEMENTS EXTRÊMES
#  Prédit séparément :
#    - FORTE HAUSSE (> +1.5% sur 5j)
#    - FORTE BAISSE (< -1.5% sur 5j)
#  Ne trade que quand le modèle est très confiant
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing      import StandardScaler
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (accuracy_score, classification_report,
                                         roc_auc_score, f1_score,
                                         confusion_matrix, precision_score,
                                         recall_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection    import TimeSeriesSplit
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
    print("✓ Optuna disponible")
except ImportError:
    OPTUNA_OK = False

os.makedirs("models", exist_ok=True)
DB_PATH  = "data/market_data.db"
N_TRIALS = 60

# Seuils de confiance — on ne trade que si très sûr
SEUIL_ACHAT = 0.55   # P(forte hausse) > 55%
SEUIL_VENTE = 0.55   # P(forte baisse) > 55%

# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────
def charger_dataset():
    conn  = sqlite3.connect(DB_PATH)
    df    = pd.read_sql("SELECT * FROM sp500_extreme_features",
                         conn, index_col='date', parse_dates=['date'])
    feats = pd.read_sql("SELECT feature FROM features_extremes",
                         conn)['feature'].tolist()
    conn.close()
    df.sort_index(inplace=True)
    feats = [f for f in feats if f in df.columns]
    return df, feats

# ─────────────────────────────────────────
# 2. NETTOYAGE
# ─────────────────────────────────────────
def preparer_X(df, feat_cols):
    cols = [c for c in feat_cols if c in df.columns]
    X    = df[cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
        q1, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col]  = X[col].clip(q1, q99)
    return X.values, cols

# ─────────────────────────────────────────
# 3. OPTUNA
# ─────────────────────────────────────────
def optimiser(X_tr, y_tr, n_trials=N_TRIALS):
    if not OPTUNA_OK:
        return None
    print(f"  Optuna : {n_trials} essais...")
    tscv = TimeSeriesSplit(n_splits=5)
    cw   = compute_class_weight('balanced', classes=np.array([0,1]), y=y_tr)
    sw   = np.where(y_tr==1, cw[1], cw[0])

    def obj(trial):
        p = {
            'n_estimators':     trial.suggest_int('n', 200, 800),
            'max_depth':        trial.suggest_int('d', 2, 7),
            'learning_rate':    trial.suggest_float('lr', 0.005, 0.15, log=True),
            'subsample':        trial.suggest_float('ss', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('cs', 0.5, 0.95),
            'min_child_weight': trial.suggest_int('mcw', 5, 60),
            'gamma':            trial.suggest_float('g', 0, 0.5),
            'reg_alpha':        trial.suggest_float('a', 0, 1),
            'reg_lambda':       trial.suggest_float('l', 0.5, 3),
        }
        sp = max(0.3, (y_tr==0).sum()/max(1,(y_tr==1).sum()))
        scores = []
        for tr_i, va_i in tscv.split(X_tr):
            if len(np.unique(y_tr[va_i])) < 2:
                continue
            sc  = StandardScaler()
            Xtr = np.nan_to_num(sc.fit_transform(X_tr[tr_i]))
            Xva = np.nan_to_num(sc.transform(X_tr[va_i]))
            m   = xgb.XGBClassifier(**p, scale_pos_weight=sp,
                                     eval_metric='auc',
                                     use_label_encoder=False,
                                     random_state=42, n_jobs=-1, verbosity=0)
            m.fit(Xtr, y_tr[tr_i], sample_weight=sw[tr_i], verbose=False)
            scores.append(roc_auc_score(y_tr[va_i], m.predict_proba(Xva)[:,1]))
        return np.mean(scores) if scores else 0.5

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    print(f"  Meilleure AUC CV : {study.best_value:.4f}")
    return {
        'n_estimators':     bp['n'],
        'max_depth':        bp['d'],
        'learning_rate':    bp['lr'],
        'subsample':        bp['ss'],
        'colsample_bytree': bp['cs'],
        'min_child_weight': bp['mcw'],
        'gamma':            bp['g'],
        'reg_alpha':        bp['a'],
        'reg_lambda':       bp['l'],
    }

# ─────────────────────────────────────────
# 4. ENTRAÎNEMENT UN MODÈLE
# ─────────────────────────────────────────
def entrainer_modele(X_tr, y_tr, X_te, y_te, feat_cols, nom):
    print(f"\n  {'─'*50}")
    print(f"  Modèle : {nom}")
    print(f"  Train : {len(X_tr)}j | Test : {len(X_te)}j")
    print(f"  Positifs train : {y_tr.mean()*100:.1f}% | test : {y_te.mean()*100:.1f}%")

    scaler = StandardScaler()
    X_tr_s = np.nan_to_num(scaler.fit_transform(X_tr))
    X_te_s = np.nan_to_num(scaler.transform(X_te))

    sp = max(0.3, (y_tr==0).sum() / max(1,(y_tr==1).sum()))
    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_tr)
    sw = np.where(y_tr==1, cw[1], cw[0])

    # Optuna
    best_p = optimiser(X_tr_s, y_tr) or {
        'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.03,
        'subsample': 0.75, 'colsample_bytree': 0.75,
        'min_child_weight': 15, 'gamma': 0.1,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
    }

    # XGBoost
    xgb_base = xgb.XGBClassifier(
        **best_p, scale_pos_weight=sp,
        eval_metric='auc', use_label_encoder=False,
        random_state=42, n_jobs=-1
    )
    xgb_base.fit(X_tr_s, y_tr, sample_weight=sw, verbose=False)

    tscv  = TimeSeriesSplit(n_splits=3)
    xgb_m = CalibratedClassifierCV(xgb_base, method='isotonic', cv=tscv)
    xgb_m.fit(X_tr_s, y_tr)

    # Random Forest
    rf_m = RandomForestClassifier(
        n_estimators=500, max_depth=best_p['max_depth'],
        min_samples_leaf=15, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_m.fit(X_tr_s, y_tr)

    # Ensemble pondéré
    p_xgb   = xgb_m.predict_proba(X_te_s)[:,1]
    p_rf    = rf_m.predict_proba(X_te_s)[:,1]
    auc_xgb = roc_auc_score(y_te, p_xgb)
    auc_rf  = roc_auc_score(y_te, p_rf)
    tot     = auc_xgb + auc_rf
    w_xgb   = auc_xgb/tot if tot>0 else 0.5
    w_rf    = auc_rf/tot  if tot>0 else 0.5
    probas  = w_xgb*p_xgb + w_rf*p_rf
    auc_ens = roc_auc_score(y_te, probas)

    print(f"  AUC → XGB:{auc_xgb:.4f} | RF:{auc_rf:.4f} | Ens:{auc_ens:.4f}")

    # Métriques à différents seuils
    print(f"\n  Analyse par seuil de confiance :")
    print(f"  {'Seuil':>6} | {'n signaux':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>6}")
    print(f"  {'─'*55}")
    for seuil in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds_s = (probas >= seuil).astype(int)
        n_sig   = preds_s.sum()
        if n_sig == 0:
            continue
        prec = precision_score(y_te, preds_s, zero_division=0)
        rec  = recall_score(y_te, preds_s, zero_division=0)
        f1   = f1_score(y_te, preds_s, zero_division=0)
        print(f"  {seuil:>6.2f} | {n_sig:>10} | {prec:>10.4f} | {rec:>8.4f} | {f1:>6.4f}")

    # Seuil optimal F1
    best_seuil, best_f1 = 0.5, 0.0
    for s in np.arange(0.40, 0.76, 0.01):
        preds_s = (probas >= s).astype(int)
        if len(np.unique(preds_s)) < 2:
            continue
        f1 = f1_score(y_te, preds_s, average='macro')
        if f1 > best_f1:
            best_f1, best_seuil = f1, s

    preds = (probas >= best_seuil).astype(int)
    acc   = accuracy_score(y_te, preds)
    print(f"\n  Seuil optimal : {best_seuil:.2f} | Accuracy : {acc:.4f} | AUC : {auc_ens:.4f}")
    print(classification_report(y_te, preds,
                                  target_names=['Non-extrême','Extrême'],
                                  zero_division=0))

    # Feature importance
    imp = pd.Series(xgb_base.feature_importances_, index=feat_cols)
    print("  Top 10 features :")
    for fname, fval in imp.nlargest(10).items():
        print(f"    {fname:<35} {fval:.4f}")

    return {
        'xgb': xgb_m, 'rf': rf_m,
        'w_xgb': w_xgb, 'w_rf': w_rf,
        'scaler': scaler, 'seuil': best_seuil,
        'auc': auc_ens, 'feats': feat_cols
    }

# ─────────────────────────────────────────
# 5. SIGNAL DU JOUR
# ─────────────────────────────────────────
def signal_du_jour(df, modele_up, modele_down):
    print("\n" + "=" * 55)
    print("  SIGNAL DU JOUR — MOUVEMENTS EXTRÊMES")
    print("=" * 55)

    last  = df.iloc[-1]
    date  = df.index[-1].strftime('%Y-%m-%d')
    prix  = last['close']

    # Proba forte hausse
    feats_up = modele_up['feats']
    X_up     = np.array([[last.get(f,0) for f in feats_up]])
    X_up_s   = np.nan_to_num(modele_up['scaler'].transform(X_up))
    p_up     = (modele_up['w_xgb'] * modele_up['xgb'].predict_proba(X_up_s)[0,1] +
                modele_up['w_rf']  * modele_up['rf'].predict_proba(X_up_s)[0,1])

    # Proba forte baisse
    feats_dn = modele_down['feats']
    X_dn     = np.array([[last.get(f,0) for f in feats_dn]])
    X_dn_s   = np.nan_to_num(modele_down['scaler'].transform(X_dn))
    p_dn     = (modele_down['w_xgb'] * modele_down['xgb'].predict_proba(X_dn_s)[0,1] +
                modele_down['w_rf']  * modele_down['rf'].predict_proba(X_dn_s)[0,1])

    print(f"  Date  : {date}")
    print(f"  Prix  : {prix:,.2f} pts")
    print(f"  VIX   : {last.get('vix_close',0):.1f}")
    print(f"  ADX   : {last.get('adx_14',0):.1f}")
    print(f"  RSI14 : {last.get('rsi_14',50):.1f}")
    print(f"  F&G   : {last.get('fg_index',50):.0f}")
    print()
    print(f"  P(FORTE HAUSSE > +1.5% / 5j) = {p_up*100:.1f}%")
    print(f"  P(FORTE BAISSE < -1.5% / 5j) = {p_dn*100:.1f}%")
    print()

    # Décision
    if p_up >= SEUIL_ACHAT and p_up > p_dn:
        signal = "ACHAT"
        emoji  = "🟢"
        detail = f"Forte hausse probable ({p_up*100:.1f}%)"
    elif p_dn >= SEUIL_VENTE and p_dn > p_up:
        signal = "VENTE"
        emoji  = "🔴"
        detail = f"Forte baisse probable ({p_dn*100:.1f}%)"
    else:
        signal = "NEUTRE"
        emoji  = "🟡"
        detail = "Pas de mouvement extrême prévu → rester en CASH"

    conf_up = "HAUTE" if p_up > 0.65 else "MOYENNE" if p_up > 0.55 else "FAIBLE"
    conf_dn = "HAUTE" if p_dn > 0.65 else "MOYENNE" if p_dn > 0.55 else "FAIBLE"

    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  SIGNAL  →  {emoji}  {signal:<8}                  │")
    print(f"  │  {detail:<41}│")
    print(f"  │                                         │")
    print(f"  │  Conf. HAUSSE : {conf_up:<6}  Conf. BAISSE : {conf_dn:<6}│")
    print(f"  └─────────────────────────────────────────┘")

    if signal == "NEUTRE":
        print(f"\n  → Aucun mouvement extrême prévu")
        print(f"     Attends P(hausse) > {SEUIL_ACHAT*100:.0f}% ou P(baisse) > {SEUIL_VENTE*100:.0f}%")

    return signal, p_up, p_dn

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MODULE 2 — MOUVEMENTS EXTRÊMES S&P500")
    print("=" * 55)

    df, feats = charger_dataset()
    print(f"Dataset : {len(df)}j × {df.shape[1]} colonnes")

    n_up   = df['target_up'].sum()
    n_down = df['target_down'].sum()
    n_neut = len(df) - n_up - n_down
    print(f"Classes : HAUSSE={n_up} ({n_up/len(df)*100:.1f}%) | "
          f"BAISSE={n_down} ({n_down/len(df)*100:.1f}%) | "
          f"NEUTRE={n_neut} ({n_neut/len(df)*100:.1f}%)")

    # Split temporel 80/20
    split    = int(len(df) * 0.80)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]
    print(f"Split : Train={len(df_train)}j | Test={len(df_test)}j")

    X_train, cols = preparer_X(df_train, feats)
    X_test,  _    = preparer_X(df_test,  feats)

    # ── Modèle FORTE HAUSSE ───────────────────────────────────
    print("\n" + "=" * 55)
    print("  MODÈLE 1 — FORTE HAUSSE (> +1.5% / 5j)")
    print("=" * 55)
    y_tr_up = df_train['target_up'].values
    y_te_up = df_test['target_up'].values
    mod_up  = entrainer_modele(X_train, y_tr_up, X_test, y_te_up, cols, "FORTE_HAUSSE")

    # ── Modèle FORTE BAISSE ───────────────────────────────────
    print("\n" + "=" * 55)
    print("  MODÈLE 2 — FORTE BAISSE (< -1.5% / 5j)")
    print("=" * 55)
    y_tr_dn = df_train['target_down'].values
    y_te_dn = df_test['target_down'].values
    mod_dn  = entrainer_modele(X_train, y_tr_dn, X_test, y_te_dn, cols, "FORTE_BAISSE")

    # Sauvegarde
    joblib.dump(mod_up, "models/modele_extreme_hausse.pkl")
    joblib.dump(mod_dn, "models/modele_extreme_baisse.pkl")
    joblib.dump({'seuil_achat': SEUIL_ACHAT, 'seuil_vente': SEUIL_VENTE,
                  'seuil_mouvement': 0.015, 'horizon': 5},
                "models/extreme_params.pkl")
    print("\n  Modèles sauvegardés ✓")

    # Signal du jour
    signal_du_jour(df, mod_up, mod_dn)

    print("\nModule 2 terminé ✓")
    print("Prochaine étape : python 7_backtest.py")