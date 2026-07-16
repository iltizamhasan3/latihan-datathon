"""
Phase 6: Focused stacking on FULL engineered features + ordinal + seed diversity
Builds on Phase 3 success (Stack_LR = 0.5869 on full features)
"""
import numpy as np, pandas as pd, json, os, time, sys, warnings, copy
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from scipy import stats

from experiments.features import get_all_features

SEED = 42; DATA_DIR = Path("data"); EXP_DIR = Path("experiments")

def load_data():
    t = pd.read_csv(DATA_DIR / "train.csv"); te = pd.read_csv(DATA_DIR / "test.csv"); s = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return t, te, s

def get_cv(rs=SEED): return RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=rs)

def evaluate_classifier(X, y, model, name, X_test=None):
    cv = get_cv(); n = len(y)
    oof_preds, oof_probs = np.zeros(n, dtype=int), np.zeros((n, 4))
    fold_m, tpreds = [], []; start = time.time()
    for fi, (tr, val) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[val])
        m = copy.deepcopy(model)
        # Handle CatBoost param conflict
        if hasattr(m, 'set_params') and hasattr(m, 'random_seed'):
            try: m.set_params(random_seed=SEED+fi)
            except: pass
        m.fit(X_tr, y[tr])
        oof_preds[val] = np.ravel(m.predict(X_val))
        if hasattr(m, 'predict_proba'): oof_probs[val] = m.predict_proba(X_val)
        tr_acc = accuracy_score(y[tr], m.predict(X_tr))
        val_acc = accuracy_score(y[val], oof_preds[val])
        fold_m.append({'fold': fi, 'accuracy': val_acc,
                       'f1': f1_score(y[val], oof_preds[val], average='macro'),
                       'bal': balanced_accuracy_score(y[val], oof_preds[val]),
                       'train_acc': tr_acc, 'gap': tr_acc - val_acc})
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
    pd.DataFrame({'id': range(len(metrics['oof_predictions'])), 'target': metrics['oof_predictions']}).to_csv(
        EXP_DIR / "oof_predictions" / "best_oof.csv", index=False)
    print(f"  >>> CHECKPOINT: {eid} {model} = {metrics.get('oof_accuracy',0):.4f}")

def make_sub(preds, template, name):
    s = template.copy(); s['target'] = np.ravel(preds).astype(int); s.to_csv(EXP_DIR / name, index=False)

# ================================================================
# ORDINAL OOF GENERATOR (reproduces Phase 2 best)
# ================================================================
def get_ordinal_oof(X, y, X_test=None):
    """Generate OOF probabilities from ordinal cumulative model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    cv = get_cv(); n = len(y)
    oof_probs = np.zeros((n, 3))
    test_list = [[], [], []]

    for fi, (tr, val) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[val])
        if X_test is not None: X_te = scaler.transform(X_test)

        for k in range(3):
            yk = (y >= (k+1)).astype(int)
            m = HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, random_state=SEED+fi)
            m.fit(X_tr, yk[tr])
            oof_probs[val, k] = m.predict_proba(X_val)[:, 1]
            if X_test is not None:
                test_list[k].append(m.predict_proba(X_te)[:, 1])

    # Convert to class probs
    class_probs = np.zeros((n, 4))
    class_probs[:, 0] = 1 - oof_probs[:, 0]
    class_probs[:, 1] = oof_probs[:, 0] - oof_probs[:, 1]
    class_probs[:, 2] = oof_probs[:, 1] - oof_probs[:, 2]
    class_probs[:, 3] = oof_probs[:, 2]
    class_probs = np.clip(class_probs, 0.001, 0.999)
    class_probs /= class_probs.sum(axis=1, keepdims=True)

    test_class_probs = None
    if X_test is not None and test_list[0]:
        test_class_probs = np.zeros((X_test.shape[0], 4))
        test_class_probs[:, 0] = 1 - np.mean(test_list[0], axis=0)
        test_class_probs[:, 1] = np.mean(test_list[0], axis=0) - np.mean(test_list[1], axis=0)
        test_class_probs[:, 2] = np.mean(test_list[1], axis=0) - np.mean(test_list[2], axis=0)
        test_class_probs[:, 3] = np.mean(test_list[2], axis=0)
        test_class_probs = np.clip(test_class_probs, 0.001, 0.999)
        test_class_probs /= test_class_probs.sum(axis=1, keepdims=True)

    return class_probs, test_class_probs


# ================================================================
# MAIN
# ================================================================
best = {'accuracy': 0.5869, 'model': 'Stack_LR', 'exp_id': 'EXP-027'}
no_improve = 0; experiment_count = 55

def main():
    global best, no_improve, experiment_count

    train, test, sample_sub = load_data()
    y = train['target'].values

    print(f"\n{'='*60}")
    print(f"PHASE 6: FOCUSED FULL-FEATURE STACKING + ORDINAL")
    print(f"Best so far: {best['accuracy']:.4f} ({best['model']})")
    print(f"Remaining: {0.70 - best['accuracy']:.4f}")
    print(f"{'='*60}")

    # Use full engineered features with PCA/cluster (Phase 3 best setup)
    X_base_df = get_all_features(train)
    X_test_df = get_all_features(test)

    # Add PCA
    from sklearn.decomposition import PCA
    for gname, cols, nc in [('wp', ['nilai_minggu_%02d'%i for i in range(1,13)], 5),
                             ('ap', ['aktivitas_hari_%02d'%i for i in range(1,17)], 6)]:
        scaler = StandardScaler()
        d = scaler.fit_transform(train[cols]); dt = scaler.transform(test[cols])
        pca = PCA(n_components=nc, random_state=SEED)
        for i, c in enumerate(pca.fit_transform(d).T):
            X_base_df[f'{gname}_{i+1}'] = c
        for i, c in enumerate(pca.transform(dt).T):
            X_test_df[f'{gname}_{i+1}'] = c

    from sklearn.cluster import KMeans
    for cname, cols, nc in [('wc', ['nilai_minggu_%02d'%i for i in range(1,13)], 5),
                             ('ac', ['aktivitas_hari_%02d'%i for i in range(1,17)], 5)]:
        scaler = StandardScaler()
        d = scaler.fit_transform(train[cols]); dt = scaler.transform(test[cols])
        km = KMeans(n_clusters=nc, random_state=SEED, n_init=10); km.fit(d)
        for i, c in enumerate(km.transform(d).T):
            X_base_df[f'{cname}_cd{i+1}'] = c
        for i, c in enumerate(km.transform(dt).T):
            X_test_df[f'{cname}_cd{i+1}'] = c

    # Key interactions (selected, not exhaustive)
    X_base_df['tryout_x_task'] = train['skor_tryout'] * X_base_df['task_completion_ratio']
    X_test_df['tryout_x_task'] = test['skor_tryout'] * X_test_df['task_completion_ratio']
    X_base_df['tugas_x_tryout'] = train['tugas_selesai'] * train['skor_tryout']
    X_test_df['tugas_x_tryout'] = test['tugas_selesai'] * test['skor_tryout']
    X_base_df['week_x_tryout'] = X_base_df['week_mean'] * train['skor_tryout']
    X_test_df['week_x_tryout'] = X_test_df['week_mean'] * test['skor_tryout']

    X = X_base_df.fillna(0).replace([np.inf,-np.inf],0).values
    X_te = X_test_df.fillna(0).replace([np.inf,-np.inf],0).values
    print(f"Features: {X.shape}")

    # ================================================================
    # 1. INDIVIDUAL MODELS (strong ones on full features)
    # ================================================================
    print("\n--- Strong individual models on FULL features ---")
    oof_probs = {}; test_probs = {}; oof_preds = {}
    base_scores = []

    import catboost as cb
    import xgboost as xgb
    import lightgbm as lgb

    models = [
        ('SVC_C50', SVC(C=50, gamma='auto', probability=True, random_state=SEED)),
        ('SVC_C100', SVC(C=100, gamma='auto', probability=True, random_state=SEED)),
        ('CatBoost', cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=SEED, verbose=0)),
        ('XGBoost', xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0)),
        ('LightGBM', lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)),
        ('HistGB', HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, random_state=SEED)),
        ('RF', RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
    ]

    for mname, model in models:
        experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
        r = evaluate_classifier(X, y, model, mname, X_test=X_te)
        log_exp(eid, best['exp_id'], f"Strong {mname} on full features", "full_features",
                mname, str(model.get_params())[:150], r,
                "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
                f"base_acc={r['oof_accuracy']:.4f}")
        oof_probs[mname] = r['oof_probabilities']
        base_scores.append((mname, r['oof_accuracy']))
        if 'test_preds' in r:
            make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                     sample_sub, f"submission_{eid}.csv")

        if r['oof_accuracy'] > best['accuracy'] + 0.002:
            best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'], model=mname, exp_id=eid)
            checkpoint(eid, mname, r, "full_features", str(model.get_params())[:100])
            no_improve = 0
        else:
            no_improve += 1

    # ================================================================
    # 2. ORDINAL OOF
    # ================================================================
    print("\n--- Ordinal cumulative model ---")
    ord_oof, ord_test = get_ordinal_oof(X, y, X_test=X_te)
    mname = 'Ordinal'
    oof_probs['Ordinal'] = ord_oof
    test_probs['Ordinal'] = ord_test
    ord_acc = accuracy_score(y, np.argmax(ord_oof, axis=1))
    ord_f1 = f1_score(y, np.argmax(ord_oof, axis=1), average='macro')
    print(f"  Ordinal                      OOF: {ord_acc:.4f} F1={ord_f1:.4f}")
    base_scores.append(('Ordinal', ord_acc))

    # ================================================================
    # 3. GET TEST PROBS FOR ALL MODELS
    # ================================================================
    print("\n--- Getting test predictions ---")
    for mname, model in models:
        scaler = StandardScaler()
        scaler.fit(X)
        m = copy.deepcopy(model)
        if hasattr(m, 'set_params') and hasattr(m, 'random_seed'):
            try: m.set_params(random_seed=SEED)
            except: pass
        m.fit(scaler.transform(X), y)
        if hasattr(m, 'predict_proba'):
            test_probs[mname] = m.predict_proba(scaler.transform(X_te))

    # ================================================================
    # 4. STACKING on full feature probs
    # ================================================================
    print(f"\n{'='*60}")
    print("STACKING (on individual model predictions)")
    print(f"{'='*60}")

    ordered_names = [n for n, s in sorted(base_scores, key=lambda x: x[1], reverse=True)]
    print(f"Ranked: {[(n, f'{s:.4f}') for n, s in sorted(base_scores, key=lambda x: x[1], reverse=True)[:6]]}")

    # Try stacking top N models (N=3,4,5,6)
    for top_n in [3, 4, 5, 6]:
        stack_names = ordered_names[:top_n]
        stack_X = np.column_stack([oof_probs[n] for n in stack_names if n in oof_probs])
        stack_te = np.column_stack([test_probs[n] for n in stack_names if n in test_probs])

        print(f"\n  Stack top {top_n}: {stack_names} (feat={stack_X.shape})")
        for meta_name, meta_model in [
            ('LR', LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
            ('LR_C01', LogisticRegression(C=0.1, solver='saga', max_iter=2000, random_state=SEED)),
            ('LR_l2', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=SEED)),
            ('Ridge', RidgeClassifier(alpha=1.0, random_state=SEED)),
        ]:
            experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
            r = evaluate_classifier(stack_X, y, meta_model, f"Stack_T{top_n}_{meta_name}", X_test=stack_te)
            log_exp(eid, best['exp_id'], f"Stack top{top_n} {meta_name}", f"stack_T{top_n}",
                    f"Stack_T{top_n}_{meta_name}", str(meta_model.get_params())[:150], r,
                    "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
                    f"stack_top{top_n}_{meta_name}={r['oof_accuracy']:.4f}")
            if r['oof_accuracy'] > best['accuracy'] + 0.002:
                best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'],
                             model=f"Stack_T{top_n}_{meta_name}", exp_id=eid)
                checkpoint(eid, f"Stack_T{top_n}_{meta_name}", r, f"stack_T{top_n}",
                           str(meta_model.get_params())[:100])
                no_improve = 0
            else: no_improve += 1
            if 'test_preds' in r:
                make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                         sample_sub, f"submission_{eid}.csv")

    # ================================================================
    # 5. ENSEMBLE WITH ORIGINAL FEATURES ADDED
    # ================================================================
    print(f"\n{'='*60}")
    print("STACKING + original features")
    print(f"{'='*60}")

    for top_n in [3, 4]:
        stack_names = ordered_names[:top_n]
        stack_X = np.column_stack([oof_probs[n] for n in stack_names if n in oof_probs] + [X])
        stack_te = np.column_stack([test_probs[n] for n in stack_names if n in test_probs] + [X_te])

        for meta_name, meta_model in [
            ('LRorig', LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
            ('Ridge_orig', RidgeClassifier(alpha=1.0, random_state=SEED)),
        ]:
            experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
            r = evaluate_classifier(stack_X, y, meta_model, f"SF_T{top_n}_{meta_name}", X_test=stack_te)
            log_exp(eid, best['exp_id'], f"Stack+Feat top{top_n} {meta_name}", f"stack+feat_T{top_n}",
                    f"SF_T{top_n}_{meta_name}", str(meta_model.get_params())[:150], r,
                    "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
                    f"sf_top{top_n}_{meta_name}={r['oof_accuracy']:.4f}")
            if r['oof_accuracy'] > best['accuracy'] + 0.002:
                best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'],
                             model=f"SF_T{top_n}_{meta_name}", exp_id=eid)
                checkpoint(eid, f"SF_T{top_n}_{meta_name}", r, f"stack+feat_T{top_n}",
                           str(meta_model.get_params())[:100])
                no_improve = 0
            else: no_improve += 1
            if 'test_preds' in r:
                make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                         sample_sub, f"submission_{eid}.csv")

    # ================================================================
    # 6. SEED AVERAGING ON BEST STACK
    # ================================================================
    print(f"\n{'='*60}")
    print("SEED AVERAGING")
    print(f"{'='*60}")

    stack_names = ordered_names[:4]
    stack_X = np.column_stack([oof_probs[n] for n in stack_names if n in oof_probs])
    stack_te = np.column_stack([test_probs[n] for n in stack_names if n in test_probs])

    for seed_val in [42, 123, 2026, 777]:
        cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed_val)
        accs = []
        for tr, val in cv2.split(stack_X, y):
            m = LogisticRegression(C=1.0, max_iter=2000, random_state=seed_val)
            m.fit(stack_X[tr], y[tr])
            accs.append(accuracy_score(y[val], m.predict(stack_X[val])))
        print(f"  Seed {seed_val}: mean={np.mean(accs):.4f} std={np.std(accs):.4f}")

    # ================================================================
    # FINAL: Save all submissions
    # ================================================================
    print(f"\n{'='*60}")
    print("GENERATING FINAL SUBMISSIONS")
    print(f"{'='*60}")

    # Ensemble multiple seeds
    all_test_probs = np.zeros((X_te.shape[0], 4))
    n_seeds = 0

    for seed_val in [42, 123, 2026]:
        # Re-fit base models - create fresh instances per seed
        for mname, m_factory in [
            ('SVC_C50', lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s)),
            ('SVC_C100', lambda s: SVC(C=100, gamma='auto', probability=True, random_state=s)),
            ('CatBoost', lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0)),
            ('XGBoost', lambda s: xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbosity=0)),
            ('LightGBM', lambda s: lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbose=-1)),
        ]:
            scaler = StandardScaler()
            scaler.fit(X)
            m = m_factory(seed_val)
            m.fit(scaler.transform(X), y)
            if hasattr(m, 'predict_proba'):
                all_test_probs += m.predict_proba(scaler.transform(X_te))
                n_seeds += 1
                print(f"  {mname}(seed={seed_val}) added")

        # Ordinal
        _, ord_test_seed = get_ordinal_oof(X, y, X_test=X_te)
        if ord_test_seed is not None:
            all_test_probs += ord_test_seed
            n_seeds += 1

    all_test_probs /= n_seeds
    all_preds = np.argmax(all_test_probs, axis=1)
    make_sub(all_preds, sample_sub, "submission_seed_avg.csv")
    print(f"\nSeed-averaged submission: {np.bincount(all_preds.astype(int))}")

    # ================================================================
    # FINAL REPORT
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 6 COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {experiment_count}")
    print(f"Best: {best['model']} = {best['accuracy']:.4f}")

    if best['accuracy'] >= 0.70:
        print("\n*** TARGET 0.70 ACHIEVED! ***")
    else:
        print(f"Remaining: {0.70 - best['accuracy']:.4f}")

    with open(EXP_DIR / "reports" / "phase6_summary.json", 'w') as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
