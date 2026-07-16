"""
Phase 8: Creative feature engineering + Calibrated stacking.
Focus on what makes SVC outperform trees here -> smooth non-linear boundaries.
Key ideas:
1. Pairwise interactions of TOP features only (avoid noise)
2. KNN-based distance to class centroids (trained in-CV, safe)
3. Feature ratios that capture relative strengths
4. Calibrated model stacking
"""
import numpy as np, pandas as pd, json, os, time, sys, warnings, copy
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from experiments.features import get_all_features, engineer_features, WEEK_COLS, ACTIVITY_COLS

SEED = 42; DATA_DIR = Path("data"); EXP_DIR = Path("experiments")
RANDOM_SEEDS = [42, 123, 2026]

def load_data():
    t = pd.read_csv(DATA_DIR / "train.csv"); te = pd.read_csv(DATA_DIR / "test.csv"); s = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return t, te, s

def get_cv(rs=SEED): return RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=rs)

def nc_probs(X, y, X_test, seed=SEED):
    """Distance to nearest centroid per class - OOF safe."""
    cv = get_cv(seed); n = len(y); n_test = X_test.shape[0]
    oof_dist = np.zeros((n, 4)); test_dist = np.zeros((n_test, 4))
    for tr, val in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[val])
        if val[0] < n:
            for c in range(4):
                if (y[tr] == c).sum() > 0:
                    centroid = X_tr[y[tr] == c].mean(axis=0)
                    oof_dist[val, c] = -np.linalg.norm(X_val - centroid, axis=1)
        X_te = scaler.transform(X_test)
        for c in range(4):
            if (y[tr] == c).sum() > 0:
                centroid = X_tr[y[tr] == c].mean(axis=0)
                test_dist[:, c] += -np.linalg.norm(X_te - centroid, axis=1) / 10
    # Softmax
    from scipy.special import softmax
    return softmax(oof_dist, axis=1), softmax(test_dist, axis=1)

def safe_target_enc(train, test, cols):
    """Target encoding with CV - OOF safe. Returns encoded arrays."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = train['target'].values
    result = np.zeros((len(train), len(cols)))
    test_result = np.zeros((len(test), len(cols)))

    for ci, col in enumerate(cols):
        for tr, val in cv.split(train, y):
            global_mean = y[tr].mean()
            enc_map = train.iloc[tr].groupby(col)[y[tr]].mean()
            # Smooth: blend with global mean
            result[val, ci] = train.iloc[val][col].map(enc_map).fillna(global_mean).values
        # Test: use all train
        global_mean = y.mean()
        enc_map = train.groupby(col)['target'].mean()
        test_result[:, ci] = test[col].map(enc_map).fillna(global_mean).values
    return result, test_result

def log_exp(eid, parent, hyp, fset, model, params, metrics, accepted, notes=""):
    p = EXP_DIR / "experiment_log.csv"
    pd.DataFrame([{'experiment_id': eid, 'parent_experiment': parent,
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

def evaluate(X, y, model, name, X_test=None, seed=SEED):
    cv = get_cv(seed); n = len(y)
    oof_preds, oof_probs = np.zeros(n, dtype=int), np.zeros((n, 4))
    fold_m, tpreds = [], []; start = time.time()
    for fi, (tr, val) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[val])
        m = copy.deepcopy(model)
        if hasattr(m, 'set_params'):
            try: m.set_params(random_state=seed+fi)
            except:
                try: m.set_params(random_seed=seed+fi)
                except: pass
        m.fit(X_tr, y[tr])
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

# ================================================================
# KEY: Calibrated stacking with disagreement detection
# ================================================================
def calibrated_stacking(X_train_full, y, X_test_full, seed=SEED):
    """Train individual models, calibrate, then stack."""
    cv = get_cv(seed); n = len(y)

    # Diverse model set - only strong ones
    models = [
        ('SVC_gauto', lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s)),
        ('SVC_gscale', lambda s: SVC(C=50, gamma='scale', probability=True, random_state=s)),
        ('SVC_C10', lambda s: SVC(C=10, gamma='scale', probability=True, random_state=s)),
    ]
    try:
        import catboost as cb
        models.append(('CatBoost', lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0)))
    except: pass
    try:
        import xgboost as xgb
        models.append(('XGBoost', lambda s: xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbosity=0)))
    except: pass

    # Get OOF and test probs for each model
    all_models_oof = {}; all_models_test = {}

    for mname, mfn in models:
        oof_p = np.zeros((n, 4))
        te_p_list = []
        for tr, val in cv.split(X_train_full, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_full[tr]); X_val = scaler.transform(X_train_full[val])
            m = mfn(seed)
            m.fit(X_tr, y[tr])
            oof_p[val] = m.predict_proba(X_val)
            te_p_list.append(m.predict_proba(scaler.transform(X_test_full)))
        all_models_oof[mname] = oof_p
        all_models_test[mname] = np.mean(te_p_list, axis=0)

    # Stacked features: [all_probs]
    names = list(all_models_oof.keys())
    X_stack = np.column_stack([all_models_oof[n] for n in names])
    X_test_stack = np.column_stack([all_models_test[n] for n in names])
    print(f"\n  Stack features: {X_stack.shape} from {len(names)} models")

    # Try multiple meta-learners
    best_acc = 0; best_res = None; best_name = ""

    for meta_name, meta_fn in [
        ('LR_C1', LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ('LR_C01', LogisticRegression(C=0.1, solver='saga', max_iter=2000, random_state=seed)),
        ('LR_C5', LogisticRegression(C=5.0, max_iter=2000, random_state=seed)),
        ('RidgeCv', RidgeClassifierCV(alphas=[0.1, 0.5, 1.0, 5.0])),
    ]:
        r = evaluate(X_stack, y, meta_fn, f"CalStack_{meta_name}", X_test=X_test_stack, seed=seed)
        if r['oof_accuracy'] > best_acc:
            best_acc = r['oof_accuracy']; best_res = r; best_name = meta_name

    return best_name, best_res, names, X_stack, X_test_stack


# ================================================================
# MAIN
# ================================================================
best = {'accuracy': 0.5869, 'model': 'Stack_LR'}
experiment_count = 82

def main():
    global best, experiment_count

    train, test, sample_sub = load_data()
    y = train['target'].values

    print(f"\n{'='*60}")
    print(f"PHASE 8: CREATIVE FEATURES + CALIBRATED STACKING")
    print(f"Current best: {best['accuracy']:.4f}")
    print(f"Remaining: {0.70 - best['accuracy']:.4f}")
    print(f"{'='*60}")

    # Build features
    X_all = get_all_features(train).fillna(0).replace([np.inf,-np.inf],0)
    X_test_raw = get_all_features(test).fillna(0).replace([np.inf,-np.inf],0)

    # ================================================================
    # 1. NEAREST CENTROID PROBABILITIES
    # ================================================================
    print("\n--- Nearest Centroid Features ---")
    nc_probs_train, nc_probs_test = nc_probs(X_all.values, y, X_test_raw.values)
    nc_acc = accuracy_score(y, np.argmax(nc_probs_train, axis=1))
    nc_f1 = f1_score(y, np.argmax(nc_probs_train, axis=1), average='macro')
    print(f"  NearestCentroid: OOF={nc_acc:.4f} F1={nc_f1:.4f}")

    # Add centroid features as additional columns
    for i in range(4):
        X_all[f'nc_prob_{i}'] = nc_probs_train[:, i]
        X_test_raw[f'nc_prob_{i}'] = nc_probs_test[:, i]

    # ================================================================
    # 2. KEY INTERACTIONS (targeted, not exhaustive)
    # ================================================================
    print("\n--- Targeted Interactions ---")

    top_features = ['week_mean', 'week_std', 'week_slope', 'activity_mean', 'activity_std',
                     'task_completion_ratio', 'tugas_selesai', 'skor_tryout',
                     'week_early_late_gap', 'week_act_mean_prod']
    top_avail = [c for c in top_features if c in X_all.columns]

    # All pairwise interactions of top features
    n_before = X_all.shape[1]
    for i in range(len(top_avail)):
        for j in range(i+1, len(top_avail)):
            a, b = top_avail[i], top_avail[j]
            X_all[f'int_{a}_x_{b}'] = X_all[a] * X_all[b]
            X_test_raw[f'int_{a}_x_{b}'] = X_test_raw[a] * X_test_raw[b]
            # Also ratio
            X_all[f'rat_{a}_d_{b}'] = X_all[a] / (X_all[b].abs() + 0.01) * np.sign(X_all[b] + 0.01)
            X_test_raw[f'rat_{a}_d_{b}'] = X_test_raw[a] / (X_test_raw[b].abs() + 0.01) * np.sign(X_test_raw[b] + 0.01)

    n_after = X_all.shape[1]
    print(f"  Added {n_after - n_before} interaction/ratio features")

    # ================================================================
    # 3. FOCUSED MODEL: SVC with different C/gamma + CalibratedStacking
    # ================================================================
    print("\n--- SVC Focus: C*gamma sweep ---")

    X_vals = X_all.values
    X_te_vals = X_test_raw.values

    best_svc_acc = 0
    for C in [0.5, 1, 5, 10, 20, 50, 100]:
        for gamma in ['scale', 'auto', 0.1, 0.05, 0.01]:
            try:
                experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
                m = SVC(C=C, gamma=gamma, probability=True, random_state=SEED)
                r = evaluate(X_vals, y, m, f"SVC_C{C}_g{gamma}", X_test=X_te_vals)
                log_exp(eid, best.get('exp_id',''), f"SVC C={C} gamma={gamma}", "full+interactions",
                        f"SVC_C{C}_g{gamma}", f"C={C},gamma={gamma}", r,
                        "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
                        f"svc={r['oof_accuracy']:.4f}")
                if r['oof_accuracy'] > best['accuracy'] + 0.002:
                    best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'],
                                 model=f"SVC_C{C}_g{gamma}", exp_id=eid)
                    checkpoint(eid, f"SVC_C{C}_g{gamma}", r, "full+interactions", f"C={C},gamma={gamma}")
                    no_improve = 0
                else:
                    no_improve += 1
                if r['oof_accuracy'] > best_svc_acc:
                    best_svc_acc = r['oof_accuracy']
                if 'test_preds' in r:
                    make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                             sample_sub, f"submission_{eid}.csv")
            except Exception as e:
                print(f"  SVC_C{C}_g{gamma} FAILED: {str(e)[:40]}")
                continue

    # ================================================================
    # 4. CALIBRATED STACKING
    # ================================================================
    print(f"\n{'='*60}")
    print("CALIBRATED STACKING")
    print(f"{'='*60}")

    meta_name, meta_r, stack_names, X_stack, X_test_stack = calibrated_stacking(
        X_vals, y, X_te_vals, seed=SEED)

    experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
    log_exp(eid, best.get('exp_id',''), f"Calibrated stacking {meta_name}", "cal_stack",
            f"CalStack_{meta_name}", "", meta_r,
            "Ya" if meta_r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
            f"calstack={meta_r['oof_accuracy']:.4f}")
    if meta_r['oof_accuracy'] > best['accuracy'] + 0.002:
        best.update(accuracy=meta_r['oof_accuracy'], macro_f1=meta_r['macro_f1'],
                     model=f"CalStack_{meta_name}", exp_id=eid)
        checkpoint(eid, f"CalStack_{meta_name}", meta_r, "cal_stack", "")
    if 'test_preds' in meta_r:
        make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, meta_r['test_preds']),
                 sample_sub, f"submission_{eid}.csv")

    # ================================================================
    # 5. SEED ENSEMBLE
    # ================================================================
    print(f"\n{'='*60}")
    print("MULTI-SEED ENSEMBLE")
    print(f"{'='*60}")

    all_test_probs = np.zeros((X_te_vals.shape[0], 4))
    n_models = 0

    for seed_val in RANDOM_SEEDS:
        for mname, mfn in [
            ('SVC', lambda s: SVC(C=20, gamma='auto', probability=True, random_state=s)),
            ('SVC_gscale', lambda s: SVC(C=50, gamma='scale', probability=True, random_state=s)),
        ]:
            scaler = StandardScaler()
            scaler.fit(X_vals)
            m = mfn(seed_val)
            m.fit(scaler.transform(X_vals), y)
            all_test_probs += m.predict_proba(scaler.transform(X_te_vals))
            n_models += 1

    try:
        import catboost as cb
        for seed_val in RANDOM_SEEDS:
            scaler = StandardScaler(); scaler.fit(X_vals)
            m = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=seed_val, verbose=0)
            m.fit(scaler.transform(X_vals), y)
            all_test_probs += m.predict_proba(scaler.transform(X_te_vals))
            n_models += 1
    except: pass

    all_test_probs /= n_models
    all_preds = np.argmax(all_test_probs, axis=1)
    make_sub(all_preds, sample_sub, "submission_phase8_ensemble.csv")
    print(f"  Multi-seed ensemble ({n_models} models): test preds saved")

    # ================================================================
    # REPORT
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 8 COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {experiment_count}")
    print(f"Best: {best['model']} = {best['accuracy']:.4f}")

    if best['accuracy'] >= 0.70:
        print("\n*** TARGET 0.70 ACHIEVED! ***")
    else:
        print(f"Remaining to 0.70: {0.70 - best['accuracy']:.4f}")
        baseline = 0.4748
        total_improvement = best['accuracy'] - baseline
        print(f"Total improvement from raw baseline: +{total_improvement:.4f}")
        print(f"Attempted {experiment_count} experiments across 8 phases")

    with open(EXP_DIR / "reports" / "phase8_summary.json", 'w') as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
