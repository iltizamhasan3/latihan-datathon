"""
Phase Final: Everything combined.
- Calibrated SVC (fixes probability calibration for stacking)
- All strong models from previous phases
- RidgeCV meta-model (auto alpha selection)
- ElasticNet meta-model (L1+L2 regularization)
- ID leakage check
"""
import numpy as np, pandas as pd, json, os, time, sys, warnings, copy
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.feature_selection import mutual_info_classif

from experiments.features import get_all_features

SEED = 42; DATA_DIR = Path("data"); EXP_DIR = Path("experiments")
RSEEDS = [42, 123, 2026, 777, 31415]

def load_data():
    t = pd.read_csv(DATA_DIR / "train.csv"); te = pd.read_csv(DATA_DIR / "test.csv"); s = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return t, te, s

def get_cv(rs=SEED): return RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=rs)

def evaluate(X, y, model, name, X_test=None, seed=SEED):
    cv = get_cv(seed); n = len(y)
    oof_preds, oof_probs = np.zeros(n, dtype=int), np.zeros((n, 4))
    fold_m, tpreds = [], []; start = time.time()
    for fi, (tr, val) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[val])
        m = copy.deepcopy(model)
        try: m.fit(X_tr, y[tr])
        except: model.fit(X_tr, y[tr]); m = model
        oof_preds[val] = np.ravel(m.predict(X_val))
        if hasattr(m, 'predict_proba'): oof_probs[val] = m.predict_proba(X_val)
        tr_acc = accuracy_score(y[tr], m.predict(X_tr))
        val_acc = accuracy_score(y[val], oof_preds[val])
        fold_m.append({'fold':fi,'accuracy':val_acc,'f1':f1_score(y[val], oof_preds[val], average='macro'),
                       'bal':balanced_accuracy_score(y[val], oof_preds[val]),'train_acc':tr_acc,'gap':tr_acc-val_acc})
        if X_test is not None: tpreds.append(np.ravel(m.predict(scaler.transform(X_test))))
    el = time.time() - start
    accs = [m['accuracy'] for m in fold_m]; oof_acc = accuracy_score(y, oof_preds); oof_f1 = f1_score(y, oof_preds, average='macro')
    r = {'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)),
         'min_fold': float(np.min(accs)), 'max_fold': float(np.max(accs)), 'macro_f1': float(oof_f1),
         'balanced_accuracy': float(np.mean([m['bal'] for m in fold_m])),
         'train_score': float(np.mean([m['train_acc'] for m in fold_m])),
         'overfit_gap': float(np.mean([m['gap'] for m in fold_m])), 'runtime': el,
         'oof_predictions': oof_preds, 'oof_probabilities': oof_probs, 'oof_accuracy': float(oof_acc)}
    if tpreds: r['test_preds'] = np.array(tpreds)
    print(f"  {name:35s} OOF: {oof_acc:.4f} (±{np.std(accs):.4f}) F1={oof_f1:.4f} min={np.min(accs):.4f} gap={r['overfit_gap']:.3f} {el:.0f}s")
    return r

def log_exp(eid, hyp, fset, model, params, metrics, accepted, notes=""):
    p = EXP_DIR / "experiment_log.csv"
    pd.DataFrame([{'experiment_id': eid, 'parent_experiment': 'FINAL',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'hypothesis': hyp[:100],
        'feature_set': fset, 'model': model, 'parameters': str(params)[:150],
        'cv_strategy': "RSKF(5,2)", 'seed': SEED,
        'mean_accuracy': f"{metrics.get('oof_accuracy',0):.6f}",
        'std_accuracy': f"{metrics['std_accuracy']:.6f}",
        'minimum_fold': f"{metrics['min_fold']:.6f}",
        'macro_f1': f"{metrics['macro_f1']:.6f}",
        'balanced_accuracy': f"{metrics['balanced_accuracy']:.6f}",
        'train_score': f"{metrics['train_score']:.6f}",
        'overfit_gap': f"{metrics['overfit_gap']:.6f}",
        'runtime': f"{metrics['runtime']:.2f}", 'accepted': accepted, 'notes': notes[:200]}]
    ).to_csv(p, mode='a', header=not p.exists(), index=False)

def checkpoint(eid, model, metrics, fset, params):
    c = {'experiment_id': eid, 'model': model, 'feature_set': fset, 'parameters': params,
         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
         'metrics': {k: metrics.get(k) for k in ['mean_accuracy','std_accuracy','min_fold','macro_f1','balanced_accuracy','overfit_gap']}}
    with open(EXP_DIR / "best_config.json", 'w') as f: json.dump(c, f, indent=2)
    np.save(EXP_DIR / "oof_predictions" / "best_oof_preds.npy", metrics['oof_predictions'])
    np.save(EXP_DIR / "oof_predictions" / "best_oof_probs.npy", metrics['oof_probabilities'])
    pd.DataFrame({'id': range(len(metrics['oof_predictions'])), 'target': metrics['oof_predictions']}).to_csv(EXP_DIR / "oof_predictions" / "best_oof.csv", index=False)
    print(f"  >>> CHECKPOINT: {eid} {model} = {metrics.get('oof_accuracy',0):.4f}")

def make_sub(preds, template, name):
    s = template.copy(); s['target'] = np.ravel(preds).astype(int); s.to_csv(EXP_DIR / name, index=False)

# ================================================================
# MAIN
# ================================================================
best = {'accuracy': 0.5869, 'model': 'Stack_LR'}
experiment_count = 117

def main():
    global best, experiment_count

    train, test, sample_sub = load_data()
    y = train['target'].values

    print(f"\n{'='*60}")
    print(f"PHASE FINAL: ALL-IN STACKING + CALIBRATION")
    print(f"Current best: {best['accuracy']:.4f}")
    print(f"Remaining: {0.70 - best['accuracy']:.4f}")
    print(f"{'='*60}")

    # Load features (clean engineered set)
    X_df = get_all_features(train).fillna(0).replace([np.inf,-np.inf],0)
    X_te_df = get_all_features(test).fillna(0).replace([np.inf,-np.inf],0)
    X, X_te = X_df.values, X_te_df.values
    print(f"Features: {X.shape}")

    # ================================================================
    # 1. ID LEAKAGE CHECK
    # ================================================================
    print(f"\n{'='*60}")
    print("LEAKAGE CHECK")
    print(f"{'='*60}")

    # ID sorted check - does target have trend when sorted by ID?
    id_sorted = train.sort_values('id')
    rolling = id_sorted['target'].rolling(window=200).mean()
    print(f"  Target rolling mean (sorted by ID, w=200):")
    print(f"    First 3 windows: {rolling.iloc[199]:.4f}, {rolling.iloc[399]:.4f}, {rolling.iloc[599]:.4f}")
    print(f"    Last 3 windows:  {rolling.iloc[2600]:.4f}, {rolling.iloc[2800]:.4f}, {rolling.iloc[2999]:.4f}")

    # ID as feature
    id_mi = mutual_info_classif(train[['id']], y, random_state=SEED)[0]
    print(f"  MI(ID, target): {id_mi:.6f}")
    print(f"  MI comparison (avg feature): {np.mean(mutual_info_classif(X, y, random_state=SEED)):.6f}")

    # Check sort order leakage
    np.random.seed(SEED)
    shuffle_scores = []
    for _ in range(100):
        shuffled = y.copy()
        np.random.shuffle(shuffled)
        # Look at consecutive agreement
        shuffle_scores.append(np.mean(shuffled[:-1] == shuffled[1:]))
    orig_score = np.mean(y[:-1] == y[1:])
    print(f"  Consecutive agreement (orig): {orig_score:.4f}")
    print(f"  Consecutive agreement (shuffled avg): {np.mean(shuffle_scores):.4f} (std={np.std(shuffle_scores):.4f})")
    print(f"  -> {'POSSIBLE LEAKAGE' if orig_score > np.mean(shuffle_scores) + 3*np.std(shuffle_scores) else 'No consecutive leakage detected'}")

    # ================================================================
    # 2. CALIBRATED SVC (probability calibration via cross-val)
    # ================================================================
    print(f"\n{'='*60}")
    print("CALIBRATED SVC")
    print(f"{'='*60}")

    for C in [10, 20, 50]:
        experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
        # Use CalibratedClassifierCV to get better probabilities
        svc = SVC(C=C, gamma='auto', probability=False, random_state=SEED, cache_size=500)
        cal_svc = CalibratedClassifierCV(svc, method='sigmoid', cv=5)
        r = evaluate(X, y, cal_svc, f"CalSVC_C{C}", X_test=X_te)
        log_exp(eid, f"Calibrated SVC C={C}", "full_eng",
                f"CalSVC_C{C}", "CalibratedClassifierCV+SVC", r,
                "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
                f"calsvc={r['oof_accuracy']:.4f}")
        if r['oof_accuracy'] > best['accuracy'] + 0.002:
            best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'], model=f"CalSVC_C{C}", exp_id=eid)
            checkpoint(eid, f"CalSVC_C{C}", r, "full_eng", f"C={C}")
        if 'test_preds' in r:
            make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                     sample_sub, f"submission_{eid}.csv")

    # ================================================================
    # 3. BUILD BEST 4 MODELS FOR STACKING
    # ================================================================
    print(f"\n{'='*60}")
    print("STACKING ENSEMBLE (final version)")
    print(f"{'='*60}")

    import catboost as cb
    import xgboost as xgb
    import lightgbm as lgb

    cv = get_cv(); n = len(y)
    all_models = {}
    test_probs = {}

    # SVC Calibrated
    svc_base = SVC(C=50, gamma='auto', probability=False, random_state=SEED, cache_size=500)
    cal = CalibratedClassifierCV(svc_base, method='sigmoid', cv=5)
    svc_oof = np.zeros((n, 4)); svc_te = []
    for tr, val in cv.split(X, y):
        scaler = StandardScaler()
        m = CalibratedClassifierCV(copy.deepcopy(svc_base), method='sigmoid', cv=5)
        m.fit(scaler.fit_transform(X[tr]), y[tr])
        svc_oof[val] = m.predict_proba(scaler.transform(X[val]))
        svc_te.append(m.predict_proba(scaler.transform(X_te)))
    all_models['CalSVC'] = svc_oof
    test_probs['CalSVC'] = np.mean(svc_te, axis=0)
    print(f"  CalSVC OOF computed")

    # CatBoost
    cb_oof = np.zeros((n, 4)); cb_te = []
    for tr, val in cv.split(X, y):
        scaler = StandardScaler()
        m = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=SEED, verbose=0)
        m.fit(scaler.fit_transform(X[tr]), y[tr])
        cb_oof[val] = m.predict_proba(scaler.transform(X[val]))
        cb_te.append(m.predict_proba(scaler.transform(X_te)))
    all_models['CatBoost'] = cb_oof
    test_probs['CatBoost'] = np.mean(cb_te, axis=0)
    print(f"  CatBoost OOF computed")

    # XGBoost
    xgb_oof = np.zeros((n, 4)); xgb_te = []
    for tr, val in cv.split(X, y):
        scaler = StandardScaler()
        m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0)
        m.fit(scaler.fit_transform(X[tr]), y[tr])
        xgb_oof[val] = m.predict_proba(scaler.transform(X[val]))
        xgb_te.append(m.predict_proba(scaler.transform(X_te)))
    all_models['XGBoost'] = xgb_oof
    test_probs['XGBoost'] = np.mean(xgb_te, axis=0)
    print(f"  XGBoost OOF computed")

    # LightGBM
    lgb_oof = np.zeros((n, 4)); lgb_te = []
    for tr, val in cv.split(X, y):
        scaler = StandardScaler()
        m = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(scaler.fit_transform(X[tr]), y[tr])
        lgb_oof[val] = m.predict_proba(scaler.transform(X[val]))
        lgb_te.append(m.predict_proba(scaler.transform(X_te)))
    all_models['LightGBM'] = lgb_oof
    test_probs['LightGBM'] = np.mean(lgb_te, axis=0)
    print(f"  LightGBM OOF computed")

    # Ordinal cumulative
    from sklearn.ensemble import HistGradientBoostingClassifier
    ord_oof = np.zeros((n, 3)); ord_te = [[], [], []]
    for fi, (tr, val) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[val])
        for k in range(3):
            yk = (y >= (k+1)).astype(int)
            m = HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, random_state=SEED)
            m.fit(X_tr, yk[tr])
            ord_oof[val, k] = m.predict_proba(X_val)[:, 1]
            ord_te[k].append(m.predict_proba(scaler.transform(X_te))[:, 1])
    ord_probs = np.zeros((n, 4))
    ord_probs[:, 0] = 1 - ord_oof[:, 0]
    ord_probs[:, 1] = ord_oof[:, 0] - ord_oof[:, 1]
    ord_probs[:, 2] = ord_oof[:, 1] - ord_oof[:, 2]
    ord_probs[:, 3] = ord_oof[:, 2]
    ord_probs = np.clip(ord_probs, 0.001, 0.999)
    ord_probs /= ord_probs.sum(axis=1, keepdims=True)
    all_models['Ordinal'] = ord_probs

    ord_te_probs = np.zeros((X_te.shape[0], 4))
    ord_te_probs[:, 0] = 1 - np.mean(ord_te[0], axis=0)
    ord_te_probs[:, 1] = np.mean(ord_te[0], axis=0) - np.mean(ord_te[1], axis=0)
    ord_te_probs[:, 2] = np.mean(ord_te[1], axis=0) - np.mean(ord_te[2], axis=0)
    ord_te_probs[:, 3] = np.mean(ord_te[2], axis=0)
    ord_te_probs = np.clip(ord_te_probs, 0.001, 0.999)
    ord_te_probs /= ord_te_probs.sum(axis=1, keepdims=True)
    test_probs['Ordinal'] = ord_te_probs
    print(f"  Ordinal OOF computed")

    # ================================================================
    # 4. META-MODEL: Grid search best meta-learner
    # ================================================================
    names = list(all_models.keys())
    X_stack = np.column_stack([all_models[n] for n in names])
    X_test_stack = np.column_stack([test_probs[n] for n in names])
    print(f"\n  Stack features: {X_stack.shape} from {len(names)} models")

    best_stack = 0; best_meta_res = None; best_meta_name = ""

    meta_candidates = [
        ('RidgeCV', RidgeClassifierCV(alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])),
        ('Ridge1', RidgeClassifier(alpha=1.0)),
        ('LR_C1', LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        ('LR_C01', LogisticRegression(C=0.1, max_iter=2000, random_state=SEED)),
        ('LR_C5', LogisticRegression(C=5.0, max_iter=2000, random_state=SEED)),
        ('LR_elastic', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.5, max_iter=2000, random_state=SEED)),
        ('LR_l2', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=SEED)),
        ('LR_l1', LogisticRegression(C=1.0, penalty='l1', solver='saga', max_iter=2000, random_state=SEED)),
    ]

    for mname, meta in meta_candidates:
        experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
        r = evaluate(X_stack, y, meta, f"Final_{mname}", X_test=X_test_stack)
        log_exp(eid, f"Final stacking {mname} on {len(names)} models", f"stack_{len(names)}m",
                f"Final_{mname}", str(meta.get_params())[:150], r,
                "Ya" if r['oof_accuracy'] > best_stack+0.001 else "Tidak",
                f"stack={r['oof_accuracy']:.4f}")
        if r['oof_accuracy'] > best_stack:
            best_stack = r['oof_accuracy']; best_meta_res = r; best_meta_name = mname
        if r['oof_accuracy'] > best['accuracy'] + 0.002:
            best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'], model=f"Final_{mname}", exp_id=eid)
            checkpoint(eid, f"Final_{mname}", r, f"stack_{len(names)}m", str(meta.get_params())[:100])
        if 'test_preds' in r:
            make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                     sample_sub, f"submission_{eid}.csv")

    # ================================================================
    # 5. OPTIMAL WEIGHTED BLEND
    # ================================================================
    print(f"\n{'='*60}")
    print("WEIGHTED BLEND SEARCH")
    print(f"{'='*60}")

    best_blend = 0; best_weights = None
    # Search weight combinations for top models
    for w1 in np.arange(0, 1.1, 0.1):
        for w2 in np.arange(0, 1.1-w1, 0.1):
            w3 = 1 - w1 - w2
            if w3 < 0: continue
            # CalSVC + CatBoost + Ordinal
            blend = w1 * all_models['CalSVC'] + w2 * all_models['CatBoost'] + w3 * all_models['Ordinal']
            acc = accuracy_score(y, np.argmax(blend, axis=1))
            if acc > best_blend:
                best_blend = acc; best_weights = (w1, w2, w3)

    print(f"  Best 3-way (CalSVC+CatBoost+Ord): {best_weights} -> {best_blend:.4f}")

    # Try 4-way
    for w1 in np.arange(0, 1.1, 0.1):
        for w2 in np.arange(0, 1.1-w1, 0.1):
            for w3 in np.arange(0, 1.1-w1-w2, 0.1):
                w4 = 1 - w1 - w2 - w3
                if w4 < 0: continue
                blend = w1*all_models['CalSVC'] + w2*all_models['CatBoost'] + w3*all_models['Ordinal'] + w4*all_models['XGBoost']
                acc = accuracy_score(y, np.argmax(blend, axis=1))
                if acc > best_blend:
                    best_blend = acc; best_weights = (w1, w2, w3, w4)

    print(f"  Best 4-way (all models): {best_weights} -> {best_blend:.4f}")

    if best_blend > best['accuracy'] + 0.002:
        best.update(accuracy=best_blend, model=f"Blend_opt")
        print(f"  >>> NEW BEST from blend!")

    # ================================================================
    # 6. MULTI-SEED FINAL SUBMISSION
    # ================================================================
    print(f"\n{'='*60}")
    print("FINAL SUBMISSION (multi-seed ensemble)")
    print(f"{'='*60}")

    final_test_probs = np.zeros((X_te.shape[0], 4))
    n_models = 0

    for seed_val in [42, 123, 2026]:
        # SVC calibrated
        svc = SVC(C=50, gamma='auto', probability=False, random_state=seed_val, cache_size=500)
        cal = CalibratedClassifierCV(svc, method='sigmoid', cv=5)
        scaler = StandardScaler()
        cal.fit(scaler.fit_transform(X), y)
        final_test_probs += cal.predict_proba(scaler.transform(X_te))
        n_models += 1

        # CatBoost
        cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=seed_val, verbose=0)
        scaler = StandardScaler()
        cb_m.fit(scaler.fit_transform(X), y)
        final_test_probs += cb_m.predict_proba(scaler.transform(X_te))
        n_models += 1

    # Ordinal
    _, ord_te_final = train_ordinal_full(X, y, X_te)
    final_test_probs += ord_te_final
    n_models += 1

    final_test_probs /= n_models
    final_preds = np.argmax(final_test_probs, axis=1)
    make_sub(final_preds, sample_sub, "submission_final.csv")
    print(f"  Final submission: {np.bincount(final_preds.astype(int))}")
    print(f"  ({n_models} models averaged across seeds)")

    # ================================================================
    # REPORT
    # ================================================================
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(f"Total experiments: {experiment_count}")
    print(f"Best stacking meta: {best_meta_name} = {best_stack:.4f}")
    print(f"Best blend OOF: {best_blend:.4f}")
    print(f"Overall best: {best['model']} = {best['accuracy']:.4f}")

    baseline = 0.4748
    print(f"\nImprovement from raw baseline ({baseline}): +{best['accuracy'] - baseline:.4f}")

    if best['accuracy'] >= 0.70:
        print("\n*** TARGET 0.70 ACHIEVED! ***")
    else:
        print(f"\nBest achieved: {best['accuracy']:.4f}")
        print(f"Target: 0.7000")
        print(f"Gap: {0.70 - best['accuracy']:.4f}")
        print(f"\nGiven {experiment_count} experiments across 8+ approaches,")
        print(f"this appears to be the realistic performance ceiling for this dataset.")

    with open(EXP_DIR / "reports" / "final_report.json", 'w') as f:
        json.dump({'best_accuracy': best['accuracy'], 'best_model': best['model'],
                   'total_experiments': experiment_count, 'target_achieved': best['accuracy'] >= 0.70,
                   'baseline': baseline, 'improvement': best['accuracy'] - baseline}, f, indent=2)


def train_ordinal_full(X, y, X_test):
    """Ordinal model trained on full data for test prediction."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    n_test = X_test.shape[0]
    te_probs = np.zeros((n_test, 3))

    for k in range(3):
        yk = (y >= (k+1)).astype(int)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        te_k = []
        for tr, val in cv.split(X, yk):
            scaler = StandardScaler()
            m = HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, random_state=SEED)
            m.fit(scaler.fit_transform(X[tr]), yk[tr])
            te_k.append(m.predict_proba(scaler.transform(X_test))[:, 1])
        te_probs[:, k] = np.mean(te_k, axis=0)

    te_class = np.zeros((n_test, 4))
    te_class[:, 0] = 1 - te_probs[:, 0]
    te_class[:, 1] = te_probs[:, 0] - te_probs[:, 1]
    te_class[:, 2] = te_probs[:, 1] - te_probs[:, 2]
    te_class[:, 3] = te_probs[:, 2]
    te_class = np.clip(te_class, 0.001, 0.999)
    te_class /= te_class.sum(axis=1, keepdims=True)
    return None, te_class


if __name__ == "__main__":
    main()
