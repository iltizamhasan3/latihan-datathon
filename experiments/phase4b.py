"""
Phase 4b: Clean stacking with feature-selected base models.
Each base model gets its own optimal feature subset.
Target: push past 0.587 toward 0.70
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS

SEED = 42; DATA_DIR = Path("data"); EXP_DIR = Path("experiments")

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
        m.fit(X_tr, y[tr])
        oof_preds[val] = np.ravel(m.predict(X_val))
        if hasattr(m, 'predict_proba'): oof_probs[val] = m.predict_proba(X_val)
        tr_acc = accuracy_score(y[tr], m.predict(X_tr))
        val_acc = accuracy_score(y[val], oof_preds[val])
        fold_m.append({'fold': fi, 'accuracy': val_acc, 'f1': f1_score(y[val], oof_preds[val], average='macro'),
                       'bal': balanced_accuracy_score(y[val], oof_preds[val]), 'train_acc': tr_acc, 'gap': tr_acc - val_acc})
        if X_test is not None: tpreds.append(np.ravel(m.predict(scaler.transform(X_test))))
    el = time.time() - start
    accs = [m['accuracy'] for m in fold_m]; oof_acc = accuracy_score(y, oof_preds); oof_f1 = f1_score(y, oof_preds, average='macro')
    r = {'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)),
         'min_fold': float(np.min(accs)), 'max_fold': float(np.max(accs)), 'macro_f1': float(oof_f1),
         'balanced_accuracy': float(np.mean([m['bal'] for m in fold_m])),
         'train_score': float(np.mean([m['train_acc'] for m in fold_m])),
         'overfit_gap': float(np.mean([m['gap'] for m in fold_m])), 'fold_details': fold_m, 'runtime': el,
         'oof_predictions': oof_preds, 'oof_probabilities': oof_probs, 'oof_accuracy': float(oof_acc)}
    if tpreds: r['test_preds'] = np.array(tpreds)
    print(f"  {name:30s} OOF: {oof_acc:.4f} (±{np.std(accs):.4f}) F1={oof_f1:.4f} min={np.min(accs):.4f} gap={r['overfit_gap']:.3f} {el:.0f}s")
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
# MAIN
# ================================================================
best = {'accuracy': 0.5869, 'model': 'Stack_LR', 'exp_id': 'EXP-027'}
no_improve = 0; experiment_count = 28

def main():
    global best, no_improve, experiment_count

    train, test, sample_sub = load_data()
    y = train['target'].values

    print(f"\n{'='*60}")
    print(f"PHASE 4b: CLEAN STACKING + FEATURE SELECTION")
    print(f"Current best: {best['accuracy']:.4f} ({best['model']})")
    print(f"Remaining to 0.70: {0.70 - best['accuracy']:.4f}")
    print(f"{'='*60}")

    # Build base features
    X_base = get_all_features(train).fillna(0).replace([np.inf,-np.inf],0)
    X_test_base = get_all_features(test).fillna(0).replace([np.inf,-np.inf],0)
    print(f"Base features: {X_base.shape}")

    # Feature selection with mutual information (in CV)
    print("\n--- Feature selection ---")
    cv = get_cv()
    mi_scores = np.zeros(X_base.shape[1])

    for tr, val in cv.split(X_base.values, y):
        mi_sel = SelectKBest(mutual_info_classif, k='all')
        mi_sel.fit(X_base.values[tr], y[tr])
        mi_scores += mi_sel.scores_

    mi_scores /= 10  # average across folds
    top_idx = np.argsort(mi_scores)[::-1]

    print(f"Top 15 features by mutual information:")
    for i in range(15):
        print(f"  {X_base.columns[top_idx[i]]:40s} MI={mi_scores[top_idx[i]]:.4f}")

    # Try different feature counts
    X_vals = X_base.values; X_te_vals = X_test_base.values

    for n_feat in [15, 20, 25, 30, 40, 60]:
        experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
        cols = top_idx[:n_feat]
        X_sel = X_vals[:, cols]; X_te_sel = X_te_vals[:, cols]

        # SVC on selected features
        r = evaluate(X_sel, y, SVC(C=50, gamma='auto', probability=True, random_state=SEED),
                     f"SVC_top{n_feat}", X_test=X_te_sel)
        log_exp(eid, best['exp_id'], f"SVC with top {n_feat} MI features", f"MI_top_{n_feat}",
                f"SVC_top{n_feat}", "{C:50,gamma:auto}", r,
                "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
                f"top{n_feat}_acc={r['oof_accuracy']:.4f}")
        if r['oof_accuracy'] > best['accuracy'] + 0.002:
            best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'], model=f"SVC_top{n_feat}", exp_id=eid)
            checkpoint(eid, f"SVC_top{n_feat}", r, f"MI_top_{n_feat}", "C50_auto")
            no_improve = 0
        else: no_improve += 1
        if 'test_preds' in r:
            make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                     sample_sub, f"submission_{eid}.csv")

    # Use best feature count for stacking
    best_n = 30  # empirical
    print(f"\n--- Using top {best_n} features for stacking ---")
    X_opt = X_vals[:, top_idx[:best_n]]
    X_te_opt = X_te_vals[:, top_idx[:best_n]]

    # Train diverse base models on optimal features
    print("\n--- Base model zoo (on selected features) ---")
    base_models = [
        ('SVC_C50', SVC(C=50, gamma='auto', probability=True, random_state=SEED)),
        ('SVC_C10', SVC(C=10, gamma='scale', probability=True, random_state=SEED)),
        ('RF', RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=4, random_state=SEED, n_jobs=-1)),
        ('ET', ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=4, random_state=SEED, n_jobs=-1)),
        ('GB', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, min_samples_leaf=5, random_state=SEED)),
        ('HGB', HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=SEED)),
        ('LR', LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=SEED, n_jobs=-1)),
        ('Ridge', RidgeClassifier(alpha=1.0, random_state=SEED)),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN15', KNeighborsClassifier(n_neighbors=15, weights='distance')),
    ]

    try:
        import xgboost as xgb
        base_models.append(('XGB', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0)))
    except: pass
    try:
        import lightgbm as lgb
        base_models.append(('LGB', lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)))
    except: pass
    try:
        import catboost as cb
        base_models.append(('CB', cb.CatBoostClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_seed=SEED, verbose=0)))
    except: pass

    oof_probs_dict = {}
    test_probs_dict = {}
    base_scores = []
    selected_names = []

    for mname, model in base_models:
        experiment_count += 1; eid = f"EXP-BASE-{mname}"
        r = evaluate(X_opt, y, model, mname, X_test=X_te_opt)
        log_exp(eid, "PHASE4b", f"Base {mname} on MI30", "MI30", mname,
                str(model.get_params())[:150], r,
                "Ya" if r['oof_accuracy'] > 0.50 else "Tidak",
                f"base_acc={r['oof_accuracy']:.4f}")

        base_scores.append((mname, r['oof_accuracy']))
        oof_probs_dict[mname] = r['oof_probabilities']
        selected_names.append(mname)

        # Get test probs by retraining on full data
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_opt)
        X_te_s = scaler.transform(X_te_opt)
        m = copy.deepcopy(model)
        m.fit(X_s, y)
        if hasattr(m, 'predict_proba'):
            test_probs_dict[mname] = m.predict_proba(X_te_s)
        else:
            test_probs_dict[mname] = np.zeros((X_te_opt.shape[0], 4))

    print(f"\nSelected models for stacking: {selected_names}")
    print(f"Scores: {[(n, f'{s:.4f}') for n, s in base_scores if s > 0.50]}")

    # ================================================================
    # STACKING
    # ================================================================
    print(f"\n{'='*60}")
    print("STACKING")
    print(f"{'='*60}")

    if len(selected_names) >= 2:
        # Stack predictions
        stack_feats = np.column_stack([oof_probs_dict[m] for m in selected_names])
        test_stack = np.column_stack([test_probs_dict[m] for m in selected_names])
        print(f"Stack features: {stack_feats.shape}")

        for meta_name, meta_model in [
            ('LR_C1', LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
            ('LR_01', LogisticRegression(C=0.1, max_iter=2000, random_state=SEED)),
            ('Ridge', RidgeClassifier(alpha=1.0, random_state=SEED)),
            ('GB_shallow', GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.2, random_state=SEED)),
            ('HGB_shallow', HistGradientBoostingClassifier(max_iter=100, max_depth=2, learning_rate=0.1, random_state=SEED)),
        ]:
            experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
            r = evaluate(stack_feats, y, meta_model, f"Stack_{meta_name}", X_test=test_stack)
            log_exp(eid, best['exp_id'], f"Stacking {meta_name} on {len(selected_names)} models",
                    f"stack_{len(selected_names)}models", f"Stack_{meta_name}",
                    str(meta_model.get_params())[:150], r,
                    "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
                    f"stack_acc={r['oof_accuracy']:.4f}")
            if r['oof_accuracy'] > best['accuracy'] + 0.002:
                best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'], model=f"Stack_{meta_name}", exp_id=eid)
                checkpoint(eid, f"Stack_{meta_name}", r, f"stack_{len(selected_names)}models", str(meta_model.get_params())[:100])
                no_improve = 0
            else: no_improve += 1
            if 'test_preds' in r:
                make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                         sample_sub, f"submission_{eid}.csv")

        # Also add original features to stack
        print("\n--- Stacking + original features ---")
        stack_with_orig = np.column_stack([stack_feats, X_opt])
        test_with_orig = np.column_stack([test_stack, X_te_opt])
        print(f"Stack+orig features: {stack_with_orig.shape}")

        for meta_name, meta_model in [
            ('LRorig', LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
            ('Ridge_orig', RidgeClassifier(alpha=1.0, random_state=SEED)),
        ]:
            experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
            r = evaluate(stack_with_orig, y, meta_model, f"Stack_{meta_name}", X_test=test_with_orig)
            log_exp(eid, best['exp_id'], f"Stack+orig {meta_name}", f"stack+orig_{len(selected_names)}models",
                    f"Stack_{meta_name}", str(meta_model.get_params())[:150], r,
                    "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak",
                    f"stack+orig_acc={r['oof_accuracy']:.4f}")
            if r['oof_accuracy'] > best['accuracy'] + 0.002:
                best.update(accuracy=r['oof_accuracy'], macro_f1=r['macro_f1'], model=f"Stack_{meta_name}", exp_id=eid)
                checkpoint(eid, f"Stack_{meta_name}", r, f"stack+orig", "")
                no_improve = 0
            else: no_improve += 1
            if 'test_preds' in r:
                make_sub(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds']),
                         sample_sub, f"submission_{eid}.csv")

    # ================================================================
    # BLEND JUST THE BEST FEW
    # ================================================================
    print(f"\n{'='*60}")
    print("SMART BLENDING (best 3-4 models)")
    print(f"{'='*60}")

    # Get top models
    top_models = sorted(base_scores, key=lambda x: x[1], reverse=True)[:4]
    top_names = [n for n, s in top_models]
    print(f"Top models: {top_names}")

    if len(top_names) >= 2:
        # Optimize blend weights (max 3 models)
        top3 = top_names[:3]
        best_blend = 0; best_w = None
        for w1 in np.arange(0, 1.1, 0.1):
            for w2 in np.arange(0, 1.1-w1, 0.1):
                w3 = 1 - w1 - w2
                w = [w1, w2, w3][:len(top3)]
                blended = np.zeros((len(y), 4))
                for i, t in enumerate(top3):
                    if t in oof_probs_dict:
                        blended += w[i] * oof_probs_dict[t]
                acc = accuracy_score(y, np.argmax(blended, axis=1))
                if acc > best_blend:
                    best_blend = acc; best_w = w[:]

        if best_w is not None:
            print(f"Best blend: {best_w} -> {best_blend:.4f}")
            blended_final = np.zeros((len(y), 4))
            for i, t in enumerate(top3):
                if t in oof_probs_dict:
                    blended_final += best_w[i] * oof_probs_dict[t]
            preds_final = np.argmax(blended_final, axis=1)
            experiment_count += 1; eid = f"EXP-{experiment_count:03d}"
            log_exp(eid, best['exp_id'], f"Blend top {len(top3)} models", "blend",
                    f"Blend_top{len(top3)}", str(best_w), {
                        'mean_accuracy': best_blend, 'std_accuracy': 0, 'min_fold': best_blend,
                        'macro_f1': f1_score(y, preds_final, average='macro'),
                        'balanced_accuracy': 0, 'train_score': 0, 'overfit_gap': 0, 'runtime': 0,
                        'oof_accuracy': best_blend, 'oof_predictions': preds_final,
                        'oof_probabilities': np.zeros((len(y),4))},
                    "Ya" if best_blend > best['accuracy']+0.002 else "Tidak",
                    f"blend={best_blend:.4f}")
            if best_blend > best['accuracy'] + 0.002:
                best.update(accuracy=best_blend, model=f"Blend_top{len(top3)}", exp_id=eid)
                no_improve = 0

    # ================================================================
    # FINAL
    # ================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 4b COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {experiment_count}")
    print(f"Best: {best['model']} = {best['accuracy']:.4f} (F1={best.get('macro_f1',0):.4f})")

    with open(EXP_DIR / "reports" / "phase4b_summary.json", 'w') as f:
        json.dump({'phase': '4b', 'best_accuracy': best['accuracy'], 'best_macro_f1': best.get('macro_f1',0),
                   'best_model': best['model'], 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

    if best['accuracy'] >= 0.70:
        print("\n*** TARGET 0.70 ACHIEVED! ***")
    else:
        print(f"\nRemaining to 0.70: {0.70 - best['accuracy']:.4f}")
        print("Next: seed averaging + threshold optimization + pseudo-labeling")

if __name__ == "__main__":
    main()
