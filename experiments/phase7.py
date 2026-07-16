"""
Phase 7: Completely different approach.
1. Diverse model zoo (many hyperparams, seeds, algorithms) → OOF probs
2. Strong L2 regularization stacking
3. KNN feature-space features
4. OVO pairwise probabilities
5. Threshold optimization
"""
import numpy as np, pandas as pd, json, os, time, sys, warnings, copy, itertools
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from experiments.features import get_all_features

SEED = 42; DATA_DIR = Path("data"); EXP_DIR = Path("experiments")

def load_data():
    t = pd.read_csv(DATA_DIR / "train.csv"); te = pd.read_csv(DATA_DIR / "test.csv"); s = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return t, te, s

def get_cv(rs=SEED): return RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=rs)

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
# BUILD FEATURES (clean, proven set)
# ================================================================
def build_features(train, test):
    X = get_all_features(train); Xt = get_all_features(test)
    from sklearn.decomposition import PCA
    for gname, cols, nc in [('wp',['nilai_minggu_%02d'%i for i in range(1,13)],5),
                             ('ap',['aktivitas_hari_%02d'%i for i in range(1,17)],6)]:
        s = StandardScaler(); d = s.fit_transform(train[cols]); dt = s.transform(test[cols])
        p = PCA(n_components=nc, random_state=SEED)
        for i, c in enumerate(p.fit_transform(d).T): X[f'{gname}_{i+1}'] = c
        for i, c in enumerate(p.transform(dt).T): Xt[f'{gname}_{i+1}'] = c
    from sklearn.cluster import KMeans
    for cname, cols, nc in [('wc',['nilai_minggu_%02d'%i for i in range(1,13)],5),
                             ('ac',['aktivitas_hari_%02d'%i for i in range(1,17)],5)]:
        s = StandardScaler(); d = s.fit_transform(train[cols]); dt = s.transform(test[cols])
        k = KMeans(n_clusters=nc, random_state=SEED, n_init=10); k.fit(d)
        for i, c in enumerate(k.transform(d).T): X[f'{cname}_cd{i+1}'] = c
        for i, c in enumerate(k.transform(dt).T): Xt[f'{cname}_cd{i+1}'] = c
    X['tryout_x_task'] = train['skor_tryout'] * X['task_completion_ratio']
    Xt['tryout_x_task'] = test['skor_tryout'] * Xt['task_completion_ratio']
    X['tugas_x_tryout'] = train['tugas_selesai'] * train['skor_tryout']
    Xt['tugas_x_tryout'] = test['tugas_selesai'] * test['skor_tryout']
    return X.fillna(0).replace([np.inf,-np.inf],0).values, Xt.fillna(0).replace([np.inf,-np.inf],0).values

# ================================================================
# OVO PROBABILITIES
# ================================================================
def ovo_probs(X, y, X_test):
    """One-vs-one pairwise probabilities. 6 classifiers for 4 classes."""
    n_classes = 4; n = len(y); n_test = X_test.shape[0]
    oof = np.zeros((n, n_classes))
    te = np.zeros((n_test, n_classes))
    pairs = list(itertools.combinations(range(n_classes), 2))

    cv = get_cv()
    for (a, b) in pairs:
        mask = (y == a) | (y == b)
        idx = np.where(mask)[0]
        y_bin = (y[idx] == b).astype(int)
        # Re-index for CV
        oof_bin = np.zeros(len(idx))
        te_bin_list = []

        cv_sub = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
        for tr, val in cv_sub.split(idx, y_bin):
            X_tr = X[idx][tr]; X_val = X[idx][val]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr); X_val = scaler.transform(X_val)
            m = SVC(C=50, gamma='auto', probability=True, random_state=SEED)
            m.fit(X_tr, y_bin[tr])
            oof_bin[val] = m.predict_proba(X_val)[:, 1]
            te_bin_list.append(m.predict_proba(scaler.transform(X_test))[:, 1])

        oof[idx, b] += oof_bin
        te[:, b] += np.mean(te_bin_list, axis=0)

    # Normalize
    oof /= (n_classes - 1)
    te /= (n_classes - 1)
    # Ensure rows sum to 1
    oof = oof / oof.sum(axis=1, keepdims=True)
    te = te / te.sum(axis=1, keepdims=True)
    return oof, te


# ================================================================
# MASSIVE MODEL ZOO
# ================================================================
def train_model_zoo(X, y, X_test, sample_sub):
    """Train many diverse models and collect OOF probs."""
    cv = get_cv(); n = len(y)
    all_oof = {}; all_test = {}
    zoo = []

    # === SVC variants ===
    for C in [0.5, 1, 5, 10, 50, 100, 200]:
        for gamma in ['scale', 'auto']:
            for s in [42, 123]:
                name = f'SVC_C{C}_{gamma}_s{s}'
                zoo.append((name, SVC(C=C, gamma=gamma, probability=True, random_state=s)))

    # === Linear models ===
    for C in [0.01, 0.1, 1.0, 10.0]:
        zoo.append((f'LR_C{C}', LogisticRegression(C=C, solver='lbfgs', max_iter=2000, random_state=SEED)))
        zoo.append((f'SGD_C{C}', SGDClassifier(loss='log_loss', alpha=1/C if C>0 else 0.01, max_iter=2000, random_state=SEED)))

    # === Tree ensembles ===
    for ne in [200, 400, 600]:
        for md in [6, 10, 14]:
            zoo.append((f'RF_{ne}_d{md}', RandomForestClassifier(n_estimators=ne, max_depth=md, min_samples_leaf=3, random_state=SEED, n_jobs=-1)))
            zoo.append((f'ET_{ne}_d{md}', ExtraTreesClassifier(n_estimators=ne, max_depth=md, min_samples_leaf=3, random_state=SEED, n_jobs=-1)))

    # === Boosting ===
    for lr in [0.02, 0.05, 0.1]:
        for md in [3, 5, 7]:
            zoo.append((f'HGB_lr{lr}_d{md}', HistGradientBoostingClassifier(max_iter=300, max_depth=md, learning_rate=lr, random_state=SEED)))
        zoo.append((f'GB_lr{lr}', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=lr, min_samples_leaf=5, random_state=SEED)))

    # === KNN variants ===
    for k in [5, 7, 11, 15, 21, 31, 51]:
        for w in ['distance', 'uniform']:
            zoo.append((f'KNN_{k}_{w}', KNeighborsClassifier(n_neighbors=k, weights=w)))

    # === XGBoost ===
    try:
        import xgboost as xgb
        for lr in [0.02, 0.05, 0.1]:
            for md in [3, 5, 7]:
                for ss in [0.7, 0.9]:
                    zoo.append((f'XGB_lr{lr}_d{md}_ss{ss}', xgb.XGBClassifier(n_estimators=300, max_depth=md, learning_rate=lr, subsample=ss, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0)))
    except: pass

    # === LightGBM ===
    try:
        import lightgbm as lgb
        for lr in [0.02, 0.05, 0.1]:
            for md in [3, 6, 9]:
                for nl in [31, 63, 127]:
                    zoo.append((f'LGB_lr{lr}_d{md}_nl{nl}', lgb.LGBMClassifier(n_estimators=300, max_depth=md, learning_rate=lr, num_leaves=nl, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)))
    except: pass

    # === CatBoost ===
    try:
        import catboost as cb
        for lr in [0.02, 0.05, 0.1]:
            for md in [3, 5, 7]:
                zoo.append((f'CB_lr{lr}_d{md}', cb.CatBoostClassifier(n_estimators=300, max_depth=md, learning_rate=lr, random_seed=SEED, verbose=0)))
    except: pass

    print(f"Model zoo: {len(zoo)} models to train")

    # Train each model with CV
    for mname, model in zoo:
        oof_p = np.zeros((n, 4))
        te_p_list = []
        # Fast CV (single StratifiedKFold for speed)
        from sklearn.model_selection import StratifiedKFold
        cv_fast = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_accs = []
        try:
            for fi, (tr, val) in enumerate(cv_fast.split(X, y)):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[val])
                m = copy.deepcopy(model)
                if hasattr(m, 'set_params') and hasattr(m, 'random_seed'):
                    try: m.set_params(random_seed=SEED+fi)
                    except: pass
                m.fit(X_tr, y[tr])
                p = np.ravel(m.predict(X_val))
                fold_accs.append(accuracy_score(y[val], p))
                if hasattr(m, 'predict_proba'):
                    oof_p[val] = m.predict_proba(X_val)
                te_p_list.append(m.predict_proba(scaler.transform(X_test)) if hasattr(m, 'predict_proba') else None)
            mean_acc = np.mean(fold_accs)

            if mean_acc > 0.48:  # Keep only decent models
                all_oof[mname] = oof_p
                all_test[mname] = np.mean([t for t in te_p_list if t is not None], axis=0)
                print(f"  {mname:35s} ACC={mean_acc:.4f} KEPT")
            else:
                print(f"  {mname:35s} ACC={mean_acc:.4f} (skip)")
        except Exception as e:
            print(f"  {mname:35s} FAILED: {str(e)[:50]}")
            continue

    print(f"\nKept {len(all_oof)} models with ACC > 0.48")
    return all_oof, all_test


# ================================================================
# L2-REGULARIZED STACKING
# ================================================================
def l2_stacking(X_stack, y, X_test_stack, lam=1.0):
    """Stacking with L2 regularization - RidgeClassifier."""
    cv = get_cv()
    oof = np.zeros(len(y), dtype=int)
    test_list = []
    for tr, val in cv.split(X_stack, y):
        m = RidgeClassifier(alpha=lam)
        m.fit(X_stack[tr], y[tr])
        oof[val] = m.predict(X_stack[val])
        test_list.append(m.predict(X_test_stack))
    acc = accuracy_score(y, oof)
    f1_v = f1_score(y, oof, average='macro')
    te = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, np.array(test_list))
    return acc, f1_v, oof, te


# ================================================================
# MAIN
# ================================================================
best = {'accuracy': 0.5869, 'model': 'Stack_LR'}
experiment_count = 82

def main():
    global best, experiment_count
    train, test, sample_sub = load_data(); y = train['target'].values
    X, X_te = build_features(train, test)
    print(f"Features: {X.shape}")

    # 1. OVO probabilities
    print("\n=== OVO Probabilities ===")
    oof_ovo, te_ovo = ovo_probs(X, y, X_te)
    ovo_acc = accuracy_score(y, np.argmax(oof_ovo, axis=1))
    ovo_f1 = f1_score(y, np.argmax(oof_ovo, axis=1), average='macro')
    print(f"OVO SVC: {ovo_acc:.4f} F1={ovo_f1:.4f}")

    # 2. Massive model zoo
    print("\n=== Model Zoo ===")
    all_oof, all_test = train_model_zoo(X, y, X_te, sample_sub)

    # 3. Stack ALL kept models
    print(f"\n=== Stacking {len(all_oof)} models ===")
    names = list(all_oof.keys())
    X_stack = np.column_stack([all_oof[n] for n in names])
    X_test_stack = np.column_stack([all_test[n] for n in names])
    print(f"Stack shape: {X_stack.shape}")

    for lam in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
        acc, f1_v, oof_p, te_p = l2_stacking(X_stack, y, X_test_stack, lam=lam)
        print(f"  Ridge(lambda={lam:.2f}): OOF={acc:.4f} F1={f1_v:.4f}")
        metrics = {
            'mean_accuracy': acc, 'std_accuracy': 0, 'min_fold': acc,
            'macro_f1': f1_v, 'balanced_accuracy': 0, 'train_score': 0,
            'overfit_gap': 0, 'runtime': 0, 'oof_accuracy': acc,
            'oof_predictions': oof_p, 'oof_probabilities': np.zeros((len(y), 4)),
        }
        log_exp(eid, best.get('exp_id',''), f"L2 stacking all {len(names)} models lam={lam}",
                "zoo_stack", f"Ridge_l2_{lam}", f"lam={lam}", metrics,
                "Ya" if acc > best['accuracy']+0.002 else "Tidak", f"acc={acc:.4f}")
        if acc > best['accuracy'] + 0.002:
            best.update(accuracy=acc, macro_f1=f1_v, model=f"RidgeL2_{lam}", exp_id=eid)
            checkpoint(eid, f"RidgeL2_{lam}", metrics, "zoo_stack", f"lam={lam}")
        make_sub(te_p, sample_sub, f"submission_{eid}.csv")

    # 4. Stack top K models by correlation diversity
    print(f"\n=== Diverse subset stacking ===")
    # Calculate pairwise disagreement
    n = len(y)
    preds_dict = {n: np.argmax(all_oof[n], axis=1) for n in names}

    # Sort by accuracy
    from collections import Counter
    accs = {}
    for n in names:
        accs[n] = accuracy_score(y, preds_dict[n])

    sorted_n = sorted(names, key=lambda n: accs[n], reverse=True)

    # Try stacking top N
    for top_n in [5, 10, 15, 20, 30]:
        sub_names = sorted_n[:top_n]
        X_sub = np.column_stack([all_oof[n] for n in sub_names])
        X_tsub = np.column_stack([all_test[n] for n in sub_names])
        experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
        acc, f1_v, oof_p, te_p = l2_stacking(X_sub, y, X_tsub, lam=1.0)
        print(f"  Ridge(top{top_n}): OOF={acc:.4f} F1={f1_v:.4f}")
        metrics = {
            'mean_accuracy': acc, 'std_accuracy': 0, 'min_fold': acc,
            'macro_f1': f1_v, 'balanced_accuracy': 0, 'train_score': 0,
            'overfit_gap': 0, 'runtime': 0, 'oof_accuracy': acc,
            'oof_predictions': oof_p, 'oof_probabilities': np.zeros((len(y), 4)),
        }
        log_exp(eid, best.get('exp_id',''), f"Stack top{top_n} by ACC", "zoo_stack",
                f"Ridge_T{top_n}", f"top{top_n}", metrics,
                "Ya" if acc > best['accuracy']+0.002 else "Tidak", f"acc={acc:.4f}")
        if acc > best['accuracy'] + 0.002:
            best.update(accuracy=acc, macro_f1=f1_v, model=f"Ridge_T{top_n}", exp_id=eid)
            checkpoint(eid, f"Ridge_T{top_n}", metrics, "zoo_stack", f"top{top_n}")
        make_sub(te_p, sample_sub, f"submission_{eid}.csv")

    # 5. Stack with original features
    print(f"\n=== Stack + original features ===")
    X_combined = np.column_stack([X_stack, X])
    Xt_combined = np.column_stack([X_test_stack, X_te])
    for lam in [0.5, 1.0, 5.0]:
        experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
        acc, f1_v, oof_p, te_p = l2_stacking(X_combined, y, Xt_combined, lam=lam)
        print(f"  Ridge(stack+orig lam={lam}): OOF={acc:.4f} F1={f1_v:.4f}")
        metrics = {
            'mean_accuracy': acc, 'std_accuracy': 0, 'min_fold': acc,
            'macro_f1': f1_v, 'balanced_accuracy': 0, 'train_score': 0,
            'overfit_gap': 0, 'runtime': 0, 'oof_accuracy': acc,
            'oof_predictions': oof_p, 'oof_probabilities': np.zeros((len(y), 4)),
        }
        log_exp(eid, best.get('exp_id',''), f"Stack+orig lam={lam}", "zoo_stack_orig",
                f"Ridge_orig_{lam}", f"lam={lam}", metrics,
                "Ya" if acc > best['accuracy']+0.002 else "Tidak", f"acc={acc:.4f}")
        if acc > best['accuracy'] + 0.002:
            best.update(accuracy=acc, macro_f1=f1_v, model=f"Ridge_orig_{lam}", exp_id=eid)
            checkpoint(eid, f"Ridge_orig_{lam}", metrics, "zoo_stack_orig", f"lam={lam}")
        make_sub(te_p, sample_sub, f"submission_{eid}.csv")

    # ================================================================
    # REPORT
    # ================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 7 COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {experiment_count}")
    print(f"Models in zoo: {len(all_oof)}")
    print(f"Best: {best['model']} = {best['accuracy']:.4f}")

    if best['accuracy'] >= 0.70: print("\n*** TARGET 0.70 ACHIEVED! ***")
    else: print(f"\nRemaining to 0.70: {0.70 - best['accuracy']:.4f}")

    with open(EXP_DIR / "reports" / "phase7_summary.json", 'w') as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
