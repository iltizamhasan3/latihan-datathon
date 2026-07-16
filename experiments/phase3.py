"""
Phase 3: Ensemble, Tuning, Residual Modeling, Confidence Switching
Continues from Phase 2 results (best: OrdinalCumulative ~0.5450)
"""
import numpy as np
import pandas as pd
import json, os, time, sys, warnings, traceback
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, log_loss)
from scipy import stats as scipy_stats

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS

SEED = 42
DATA_DIR = Path("data")
EXP_DIR = Path("experiments")

# =============================================================
# LOAD & HELPERS
# =============================================================
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sub

def get_cv(seed=SEED, repeats=2):
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)

def evaluate(X, y, model, name, X_test=None, seed=SEED, repeats=2):
    cv = get_cv(seed, repeats)
    n = len(y)
    oof_preds = np.zeros(n, dtype=int)
    oof_probs = np.zeros((n, 4))
    fold_metrics = []
    test_preds_list = []
    start = time.time()

    for fi, (tr, val) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_val = scaler.transform(X[val])
        y_tr, y_val = y[tr], y[val]

        import copy
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)

        p = np.ravel(m.predict(X_val))
        pp = m.predict_proba(X_val) if hasattr(m, 'predict_proba') else None
        tp = np.ravel(m.predict(X_tr))
        if pp is not None and pp.shape[1] == 4:
            oof_probs[val] = pp
        oof_preds[val] = p

        fold_metrics.append({
            'fold': fi, 'accuracy': accuracy_score(y_val, p),
            'f1': f1_score(y_val, p, average='macro'),
            'bal': balanced_accuracy_score(y_val, p),
            'train_acc': accuracy_score(y_tr, tp),
            'gap': accuracy_score(y_tr, tp) - accuracy_score(y_val, p),
        })

        if X_test is not None:
            X_te = scaler.transform(X_test)
            test_preds_list.append(np.ravel(m.predict(X_te)))

    el = time.time() - start
    accs = [m['accuracy'] for m in fold_metrics]
    oof_acc = accuracy_score(y, oof_preds)
    oof_f1 = f1_score(y, oof_preds, average='macro')

    r = {
        'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)),
        'min_fold': float(np.min(accs)), 'max_fold': float(np.max(accs)),
        'macro_f1': float(oof_f1),
        'balanced_accuracy': float(np.mean([m['bal'] for m in fold_metrics])),
        'train_score': float(np.mean([m['train_acc'] for m in fold_metrics])),
        'overfit_gap': float(np.mean([m['gap'] for m in fold_metrics])),
        'fold_details': fold_metrics, 'runtime': el,
        'oof_predictions': oof_preds, 'oof_probabilities': oof_probs,
        'oof_fold': None, 'oof_accuracy': float(oof_acc),
    }
    if test_preds_list:
        r['test_preds'] = np.array(test_preds_list)

    print(f"  {name} — OOF: {oof_acc:.4f} (±{np.std(accs):.4f}) F1={oof_f1:.4f} min={np.min(accs):.4f} gap={r['overfit_gap']:.3f} {el:.0f}s")
    return r

def log_exp(exp_id, parent, hypothesis, fset, model, params, cv_strat, seed, metrics, accepted, notes=""):
    p = EXP_DIR / "experiment_log.csv"
    row = {'experiment_id': exp_id, 'parent_experiment': parent,
           'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'hypothesis': hypothesis, 'feature_set': fset, 'model': model,
           'parameters': str(params)[:150], 'cv_strategy': cv_strat, 'seed': seed,
           'mean_accuracy': f"{metrics.get('oof_accuracy',0):.6f}",
           'std_accuracy': f"{metrics['std_accuracy']:.6f}",
           'minimum_fold': f"{metrics['min_fold']:.6f}",
           'macro_f1': f"{metrics['macro_f1']:.6f}",
           'balanced_accuracy': f"{metrics['balanced_accuracy']:.6f}",
           'train_score': f"{metrics['train_score']:.6f}",
           'overfit_gap': f"{metrics['overfit_gap']:.6f}",
           'runtime': f"{metrics['runtime']:.2f}",
           'accepted': accepted, 'notes': notes[:200]}
    df = pd.DataFrame([row])
    if p.exists(): df.to_csv(p, mode='a', header=False, index=False)
    else: df.to_csv(p, index=False)

def checkpoint(eid, model, metrics, fset, params):
    c = {'experiment_id': eid, 'model': model, 'feature_set': fset, 'parameters': params,
         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
         'metrics': {k: metrics.get(k) for k in ['mean_accuracy','std_accuracy','min_fold','macro_f1','balanced_accuracy','overfit_gap']}}
    with open(EXP_DIR / "best_config.json", 'w') as f:
        json.dump(c, f, indent=2)
    np.save(EXP_DIR / "oof_predictions" / "best_oof_preds.npy", metrics['oof_predictions'])
    np.save(EXP_DIR / "oof_predictions" / "best_oof_probs.npy", metrics['oof_probabilities'])
    pd.DataFrame({'id': range(len(metrics['oof_predictions'])), 'target': metrics['oof_predictions']}).to_csv(
        EXP_DIR / "oof_predictions" / "best_oof.csv", index=False)
    print(f"  >>> CHECKPOINT: {eid} {model} = {metrics.get('oof_accuracy',0):.4f}")

def make_sub(preds, template, name):
    s = template.copy()
    s['target'] = np.ravel(preds).astype(int)
    s.to_csv(EXP_DIR / name, index=False)

def run(exp_id, model, name, fset, X, y, X_test, sub, parent, hyp, params_s=None):
    res = evaluate(X, y, model, name, X_test=X_test, seed=SEED, repeats=2)
    acc = res['oof_accuracy']
    imp = acc - best['accuracy']
    acc_accept = "Ya" if imp >= 0.002 else ("Marginal" if imp > 0 else "Tidak")
    log_exp(exp_id, parent, hyp, fset, name, params_s or str(model.get_params() if hasattr(model,'get_params') else ''),
            "RSKF(5,2)", SEED, res, acc_accept, f"imp={imp:.4f}")
    if acc > best['accuracy'] + 0.002:
        best.update(accuracy=acc, macro_f1=res['macro_f1'], model=name, exp_id=exp_id)
        checkpoint(exp_id, name, res, fset, params_s or str(model.get_params() if hasattr(model,'get_params') else ''))
        return True, res
    elif imp > 0:
        print(f"  Marginal gain: +{imp:.4f}")
    if 'test_preds' in res:
        mp = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, res['test_preds'])
        make_sub(mp, sub, f"submission_{exp_id}.csv")
    return False, res

# =============================================================
# FEATURE BUILDING
# =============================================================
def build_advanced(train_df, test_df):
    """Build the best feature set from Phase 2."""
    base = get_all_features(train_df)
    base_t = get_all_features(test_df)

    # PCA
    for gname, cols, nc in [('wp', WEEK_COLS, 5), ('ap', ACTIVITY_COLS, 6)]:
        scaler = StandardScaler()
        d = scaler.fit_transform(train_df[cols])
        dt = scaler.transform(test_df[cols])
        pca = PCA(n_components=nc, random_state=SEED)
        for i, c in enumerate(pca.fit_transform(d).T):
            base[f'{gname}_{i+1}'] = c
        for i, c in enumerate(pca.transform(dt).T):
            base_t[f'{gname}_{i+1}'] = c

    # Clusters
    for cname, cols, nc in [('wc', WEEK_COLS, 5), ('ac', ACTIVITY_COLS, 5)]:
        scaler = StandardScaler()
        d = scaler.fit_transform(train_df[cols])
        dt = scaler.transform(test_df[cols])
        km = KMeans(n_clusters=nc, random_state=SEED, n_init=10)
        km.fit(d)
        for i, c in enumerate(km.transform(d).T):
            base[f'{cname}_cd{i+1}'] = c
        for i, c in enumerate(km.transform(dt).T):
            base_t[f'{cname}_cd{i+1}'] = c
        base[f'{cname}_lab'] = km.predict(d)
        base_t[f'{cname}_lab'] = km.predict(dt)

    base = base.fillna(0).replace([np.inf, -np.inf], 0)
    base_t = base_t.fillna(0).replace([np.inf, -np.inf], 0)
    return base.values, base_t.values

# =============================================================
# ENSEMBLE: Blending from OOF predictions
# =============================================================
def ensemble_blend(oof_dict, test_dict, y, X_test_len):
    """Find optimal blend weights from OOF predictions."""
    models = list(oof_dict.keys())
    n = len(y)
    print(f"\n  Ensemble blending: {models}")

    best_w = None
    best_acc = 0
    for w1 in np.arange(0, 1.1, 0.1):
        for w2 in np.arange(0, 1.1 - w1, 0.1):
            w3 = 1 - w1 - w2
            if len(models) == 2:
                blended = w1 * oof_dict[models[0]] + (1-w1) * oof_dict[models[1]]
            elif len(models) >= 3 and w3 >= 0:
                blended = w1 * oof_dict[models[0]] + w2 * oof_dict[models[1]] + w3 * oof_dict[models[2]]
            else:
                continue

            if blended.ndim == 1 or blended.shape[1] != 4:
                # It's class predictions, not probabilities
                continue

            preds = np.argmax(blended, axis=1)
            acc = accuracy_score(y, preds)
            if acc > best_acc:
                best_acc = acc
                best_w = (w1, w2, w3) if len(models) >= 3 else (w1, 1-w1)

    if best_w is not None:
        print(f"  Best blend: {best_w} -> OOF {best_acc:.4f}")
        if len(models) == 2:
            test_blend = best_w[0] * test_dict[models[0]] + best_w[1] * test_dict[models[1]]
        else:
            w1, w2, w3 = best_w
            test_blend = w1 * test_dict[models[0]] + w2 * test_dict[models[1]] + w3 * test_dict[models[2]]
        test_preds = np.argmax(test_blend, axis=1) if test_blend.ndim > 1 and test_blend.shape[1] == 4 else np.round(test_blend).clip(0,3).astype(int)
        return best_acc, test_preds, best_w
    return best_acc, None, best_w

# =============================================================
# MAIN
# =============================================================
best = {'accuracy': 0.5450, 'macro_f1': 0.5476, 'model': 'OrdinalCumulative',
        'feature_set': 'full_advanced', 'exp_id': 'EXP-019'}
experiment_count = 19
no_improve = 3

def main():
    global best, experiment_count, no_improve

    train, test, sample_sub = load_data()
    y = train['target'].values

    print(f"\n{'='*60}")
    print(f"PHASE 3: ENSEMBLE + TUNING + RESIDUAL + SWITCHING")
    print(f"Starting from: {best['accuracy']:.4f} ({best['model']})")
    print(f"Target: 0.70, remaining: {0.70 - best['accuracy']:.4f}")
    print(f"{'='*60}")

    # Build features
    X_adv, X_t_adv = build_advanced(train, test)
    print(f"\nAdvanced features: {X_adv.shape}")

    # =============================================================
    # EXP: CatBoost tuned + Ordinal + SVC as ensemble candidates
    # =============================================================
    print("\n--- Building strong model candidates ---")

    import catboost as cb
    candidates = []
    oof_dict = {}
    test_dict = {}
    all_test_probs = []

    # 1) CatBoost
    cb_model = cb.CatBoostClassifier(n_estimators=600, max_depth=6, learning_rate=0.03,
                                      random_seed=SEED, verbose=0)
    experiment_count += 1
    eid = f"EXP-{experiment_count:03d}"
    r_cb = evaluate(X_adv, y, cb_model, "CatBoost600", X_test=X_t_adv)
    log_exp(eid, best['exp_id'], "CatBoost with more iterations", "advanced",
            "CatBoost600", "{n_estimators:600, max_depth:6, lr:0.03}",
            "RSKF(5,2)", SEED, r_cb,
            "Ya" if r_cb['oof_accuracy'] > best['accuracy']+0.002 else "Tidak")
    if r_cb['oof_accuracy'] > best['accuracy'] + 0.002:
        best.update(accuracy=r_cb['oof_accuracy'], macro_f1=r_cb['macro_f1'],
                     model='CatBoost600', exp_id=eid)
        checkpoint(eid, 'CatBoost600', r_cb, 'advanced', 'n600_d6_lr003')
        no_improve = 0
    else:
        no_improve += 1
    candidates.append(('CatBoost', r_cb))
    oof_dict['CatBoost'] = r_cb['oof_probabilities']
    if 'test_probs' not in r_cb:
        # Compute test probs manually
        r_cb_test = []
        cv = get_cv()
        for tr, val in cv.split(X_adv, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_adv[tr])
            cb_model.fit(X_tr, y[tr])
            r_cb_test.append(cb_model.predict_proba(scaler.transform(X_t_adv)))
        test_dict['CatBoost'] = np.mean(r_cb_test, axis=0)
    else:
        test_dict['CatBoost'] = np.mean(r_cb['test_probs'], axis=0)
    if 'test_preds' in r_cb:
        make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r_cb['test_preds']), sample_sub, f"submission_{eid}.csv")

    # 2) SVC tuned
    experiment_count += 1
    eid = f"EXP-{experiment_count:03d}"
    svc = SVC(C=50, gamma='auto', probability=True, random_state=SEED)
    r_svc = evaluate(X_adv, y, svc, "SVC_C50", X_test=X_t_adv, repeats=2)
    log_exp(eid, best['exp_id'], "SVC C=50 gamma=auto", "advanced",
            "SVC_C50", "{C:50, gamma:auto}", "RSKF(5,2)", SEED, r_svc,
            "Ya" if r_svc['oof_accuracy'] > best['accuracy']+0.002 else "Tidak")
    if r_svc['oof_accuracy'] > best['accuracy'] + 0.002:
        best.update(accuracy=r_svc['oof_accuracy'], macro_f1=r_svc['macro_f1'],
                     model='SVC_C50', exp_id=eid)
        checkpoint(eid, 'SVC_C50', r_svc, 'advanced', 'C50_auto')
        no_improve = 0
    else:
        no_improve += 1
    candidates.append(('SVC', r_svc))
    oof_dict['SVC'] = r_svc['oof_probabilities']
    # Test probs for SVC
    svc_test_list = []
    cv = get_cv()
    for tr, val in cv.split(X_adv, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_adv[tr])
        svc.fit(X_tr, y[tr])
        svc_test_list.append(svc.predict_proba(scaler.transform(X_t_adv)))
    test_dict['SVC'] = np.mean(svc_test_list, axis=0)
    if 'test_preds' in r_svc:
        make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r_svc['test_preds']), sample_sub, f"submission_{eid}.csv")

    # 3) RandomForest strong
    experiment_count += 1
    eid = f"EXP-{experiment_count:03d}"
    rf = RandomForestClassifier(n_estimators=600, max_depth=16, min_samples_leaf=2,
                                 random_state=SEED, n_jobs=-1)
    r_rf = evaluate(X_adv, y, rf, "RF600", X_test=X_t_adv)
    log_exp(eid, best['exp_id'], "RandomForest stronger", "advanced",
            "RF600", "{n_estimators:600, max_depth:16}", "RSKF(5,2)", SEED, r_rf,
            "Ya" if r_rf['oof_accuracy'] > best['accuracy']+0.002 else "Tidak")
    if r_rf['oof_accuracy'] > best['accuracy'] + 0.002:
        best.update(accuracy=r_rf['oof_accuracy'], macro_f1=r_rf['macro_f1'],
                     model='RF600', exp_id=eid)
        checkpoint(eid, 'RF600', r_rf, 'advanced', 'n600_d16')
        no_improve = 0
    else:
        no_improve += 1
    candidates.append(('RF', r_rf))
    oof_dict['RF'] = r_rf['oof_probabilities']
    rf_test_list = []
    cv = get_cv()
    for tr, val in cv.split(X_adv, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_adv[tr])
        rf.fit(X_tr, y[tr])
        rf_test_list.append(rf.predict_proba(scaler.transform(X_t_adv)))
    test_dict['RF'] = np.mean(rf_test_list, axis=0)
    if 'test_preds' in r_rf:
        make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r_rf['test_preds']), sample_sub, f"submission_{eid}.csv")

    # =============================================================
    # ENSEMBLE: Blend all strong candidates
    # =============================================================
    print(f"\n{'='*60}")
    print("ENSEMBLE BLENDING")
    print(f"{'='*60}")

    # Pairwise blends
    for m1 in ['CatBoost', 'SVC', 'RF']:
        for m2 in ['CatBoost', 'SVC', 'RF']:
            if m1 >= m2: continue
            experiment_count += 1
            eid = f"EXP-{experiment_count:03d}"
            p1, p2 = oof_dict[m1], oof_dict[m2]
            best_b = 0
            for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                blend = w * p1 + (1-w) * p2
                acc = accuracy_score(y, np.argmax(blend, axis=1))
                if acc > best_b: best_b = acc
            print(f"  {m1}+{m2}: best blend={best_b:.4f}")
            log_exp(eid, best['exp_id'], f"Blend {m1}+{m2}", "ensemble",
                    f"Blend_{m1}_{m2}", "", "OOF-optimized", SEED,
                    {'oof_accuracy': best_b, 'mean_accuracy': best_b,
                     'std_accuracy': 0, 'min_fold': best_b, 'max_fold': best_b,
                     'macro_f1': 0, 'balanced_accuracy': 0, 'train_score': 0,
                     'overfit_gap': 0, 'runtime': 0,
                     'oof_predictions': np.argmax(best_b * p1 + (1-best_b) * p2 if isinstance(best_b, float) else p1, axis=1),
                     'oof_probabilities': np.zeros((len(y), 4))},
                    "Ya" if best_b > best['accuracy']+0.002 else "Tidak",
                    f"best_ blend_acc={best_b:.4f}")
            if best_b > best['accuracy'] + 0.002:
                best.update(accuracy=best_b, model=f'Blend_{m1}_{m2}', exp_id=eid)
                no_improve = 0

    # Three-way blend
    if len(oof_dict) >= 3:
        experiment_count += 1
        eid = f"EXP-{experiment_count:03d}"
        best_b3, te_preds, best_w3 = ensemble_blend(oof_dict, test_dict, y, X_t_adv.shape[0])
        print(f"  3-way blend: {best_w3} -> {best_b3:.4f}")
        log_exp(eid, best['exp_id'], "3-way blend CatBoost+SVC+RF", "ensemble",
                "Blend_3way", str(best_w3), "OOF-optimized", SEED,
                {'oof_accuracy': best_b3, 'mean_accuracy': best_b3,
                 'std_accuracy': 0, 'min_fold': best_b3, 'max_fold': best_b3,
                 'macro_f1': 0, 'balanced_accuracy': 0, 'train_score': 0,
                 'overfit_gap': 0, 'runtime': 0,
                 'oof_predictions': np.argmax(sum(w3 * oof_dict[m] for m, w3 in zip(oof_dict.keys(), best_w3) if w3 > 0), axis=1) if best_w3 else np.zeros(len(y)),
                 'oof_probabilities': np.zeros((len(y),4))},
                "Ya" if best_b3 > best['accuracy']+0.002 else "Tidak",
                f"3way blend={best_b3:.4f}")
        if best_b3 > best['accuracy'] + 0.002:
            best.update(accuracy=best_b3, model='Blend_3way', exp_id=eid)
            no_improve = 0
        if te_preds is not None:
            make_sub(te_preds, sample_sub, f"submission_{eid}.csv")

    # =============================================================
    # STACKING: Meta-model on OOF probabilities
    # =============================================================
    print(f"\n{'='*60}")
    print("STACKING (Meta-model on OOF probs)")
    print(f"{'='*60}")

    X_stack = np.column_stack([oof_dict[m] for m in ['CatBoost', 'SVC', 'RF']])
    X_test_stack = np.column_stack([test_dict[m] for m in ['CatBoost', 'SVC', 'RF']])
    print(f"  Stack features: {X_stack.shape}")

    for meta_name, meta_cls in [
        ('Stack_LR', LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ('Stack_Ridge', LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
    ]:
        experiment_count += 1
        eid = f"EXP-{experiment_count:03d}"
        r = evaluate(X_stack, y, meta_cls, meta_name, X_test=X_test_stack, seed=SEED, repeats=2)
        log_exp(eid, best['exp_id'], f"Stacking {meta_name}", "stack_meta",
                meta_name, str(meta_cls.get_params()), "RSKF(5,2)", SEED, r,
                "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak")
        if r['oof_accuracy'] > best['accuracy'] + 0.002:
            best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'],
                         model=meta_name, exp_id=eid)
            checkpoint(eid, meta_name, r, 'stack_meta', str(meta_cls.get_params()))
            no_improve = 0
        else:
            no_improve += 1
        if 'test_preds' in r:
            make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                     sample_sub, f"submission_{eid}.csv")

    # =============================================================
    # CONFIDENCE SWITCHING
    # =============================================================
    print(f"\n{'='*60}")
    print("CONFIDENCE SWITCHING")
    print(f"{'='*60}")

    experiment_count += 1
    eid = f"EXP-{experiment_count:03d}"

    cv = get_cv()
    n = len(y)
    sw_preds = np.zeros(n, dtype=int)
    sw_fold_accs = []

    for fi, (tr, val) in enumerate(cv.split(X_adv, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_adv[tr])
        X_val = scaler.transform(X_adv[val])
        y_tr, y_val = y[tr], y[val]

        # Train both CatBoost and SVC
        cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                      random_seed=SEED+fi, verbose=0)
        svc_m = SVC(C=50, gamma='auto', probability=True, random_state=SEED+fi)

        cb_m.fit(X_tr, y_tr)
        svc_m.fit(X_tr, y_tr)

        cb_prob = cb_m.predict_proba(X_val)
        svc_prob = svc_m.predict_proba(X_val)

        cb_conf = np.max(cb_prob, axis=1)
        svc_conf = np.max(svc_prob, axis=1)
        cb_pred = np.ravel(cb_m.predict(X_val))
        svc_pred = np.ravel(svc_m.predict(X_val))
        agree = (cb_pred == svc_pred)

        # Strategy: If agree, use either. If disagree, use higher-confidence model
        for i in range(len(val)):
            if agree[i]:
                sw_preds[val[i]] = cb_pred[i]
            elif cb_conf[i] > svc_conf[i] + 0.05:
                sw_preds[val[i]] = cb_pred[i]
            elif svc_conf[i] > cb_conf[i] + 0.05:
                sw_preds[val[i]] = svc_pred[i]
            else:
                # Low confidence disagreement: use blend
                blended = (cb_prob[i] + svc_prob[i]) / 2
                sw_preds[val[i]] = np.argmax(blended)

        sw_fold_accs.append(accuracy_score(y_val, sw_preds[val]))

    sw_acc = accuracy_score(y, sw_preds)
    sw_f1 = f1_score(y, sw_preds, average='macro')
    print(f"  ConfidenceSwitching — OOF: {sw_acc:.4f}, F1={sw_f1:.4f}")

    log_exp(eid, best['exp_id'], "Confidence switching CatBoost vs SVC",
            "advanced", "ConfidenceSwitch", "CatBoost+SVC",
            "RSKF(5,2)", SEED,
            {'oof_accuracy': sw_acc, 'mean_accuracy': float(np.mean(sw_fold_accs)),
             'std_accuracy': float(np.std(sw_fold_accs)),
             'min_fold': float(np.min(sw_fold_accs)), 'max_fold': float(np.max(sw_fold_accs)),
             'macro_f1': sw_f1, 'balanced_accuracy': balanced_accuracy_score(y, sw_preds),
             'train_score': 0, 'overfit_gap': 0, 'runtime': 0,
             'oof_predictions': sw_preds, 'oof_probabilities': np.zeros((n, 4))},
            "Ya" if sw_acc > best['accuracy']+0.002 else "Tidak",
            f"conf_switch={sw_acc:.4f}")
    if sw_acc > best['accuracy'] + 0.002:
        best.update(accuracy=sw_acc, macro_f1=sw_f1, model='ConfidenceSwitch', exp_id=eid)
        no_improve = 0

    # =============================================================
    # FINAL REPORT
    # =============================================================
    print(f"\n{'='*60}")
    print("PHASE 3 COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {experiment_count}")
    print(f"Best: {best['model']} = {best['accuracy']:.4f} (F1={best['macro_f1']:.4f})")
    print(f"No-improvement: {no_improve}")

    report = {
        'phase': 3, 'experiments_run': experiment_count,
        'best_accuracy': best['accuracy'], 'best_macro_f1': best['macro_f1'],
        'best_model': best['model'], 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(EXP_DIR / "reports" / "phase3_summary.json", 'w') as f:
        json.dump(report, f, indent=2)

    remaining = 0.70 - best['accuracy']
    if best['accuracy'] >= 0.70:
        print("\n*** TARGET 0.70 ACHIEVED! ***")
    else:
        print(f"\nRemaining to 0.70: {remaining:.4f}")
        if remaining < 0.02:
            print("Close! Fine-tune ensemble weights and thresholds.")
        elif remaining < 0.05:
            print("Need more: try PCA features, more interactions, pseudo-labeling.")
        else:
            print("Further improvement needed: explore deeper feature engineering.")

if __name__ == "__main__":
    main()
