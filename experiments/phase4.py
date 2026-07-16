"""
Phase 4: Aggressive Stacking + Deeper Features + Multi-model Ensemble
Target: Push from 0.587 toward 0.70
"""
import numpy as np, pandas as pd, json, os, time, sys, warnings, traceback, copy
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, SelectFromModel
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, log_loss)
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS

SEED = 42
DATA_DIR = Path("data")
EXP_DIR = Path("experiments")

def load_data():
    t = pd.read_csv(DATA_DIR / "train.csv"); te = pd.read_csv(DATA_DIR / "test.csv"); s = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return t, te, s

def get_cv(rs=SEED):
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=rs)

def evaluate(X, y, model, name, X_test=None, seed=SEED):
    cv = get_cv(seed)
    n = len(y)
    oof_preds, oof_probs = np.zeros(n, dtype=int), np.zeros((n, 4))
    fold_m = []
    tpreds_list = []
    start = time.time()

    for fi, (tr, val) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[val])
        m = copy.deepcopy(model)
        m.fit(X_tr, y[tr])
        oof_preds[val] = np.ravel(m.predict(X_val))
        if hasattr(m, 'predict_proba'):
            oof_probs[val] = m.predict_proba(X_val)
        tr_pred = accuracy_score(y[tr], m.predict(X_tr))
        val_acc = accuracy_score(y[val], oof_preds[val])
        fold_m.append({'fold': fi, 'accuracy': val_acc,
                       'f1': f1_score(y[val], oof_preds[val], average='macro'),
                       'bal': balanced_accuracy_score(y[val], oof_preds[val]),
                       'train_acc': tr_pred, 'gap': tr_pred - val_acc})
        if X_test is not None:
            tpreds_list.append(np.ravel(m.predict(scaler.transform(X_test))))

    el = time.time() - start
    accs = [m['accuracy'] for m in fold_m]
    oof_acc = accuracy_score(y, oof_preds)
    oof_f1 = f1_score(y, oof_preds, average='macro')
    r = {'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)),
         'min_fold': float(np.min(accs)), 'max_fold': float(np.max(accs)),
         'macro_f1': float(oof_f1),
         'balanced_accuracy': float(np.mean([m['bal'] for m in fold_m])),
         'train_score': float(np.mean([m['train_acc'] for m in fold_m])),
         'overfit_gap': float(np.mean([m['gap'] for m in fold_m])),
         'fold_details': fold_m, 'runtime': el,
         'oof_predictions': oof_preds, 'oof_probabilities': oof_probs,
         'oof_accuracy': float(oof_acc)}
    if tpreds_list:
        r['test_preds'] = np.array(tpreds_list)
    print(f"  {name:30s} OOF: {oof_acc:.4f} (±{np.std(accs):.4f}) F1={oof_f1:.4f} min={np.min(accs):.4f} gap={r['overfit_gap']:.3f} {el:.0f}s")
    return r

def log_exp(eid, parent, hypothesis, fset, model, params, seed, metrics, accepted, notes=""):
    p = EXP_DIR / "experiment_log.csv"
    row = {'experiment_id': eid, 'parent_experiment': parent,
           'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'hypothesis': hypothesis[:100], 'feature_set': fset, 'model': model,
           'parameters': str(params)[:150], 'cv_strategy': "RSKF(5,2)", 'seed': seed,
           'mean_accuracy': f"{metrics.get('oof_accuracy',0):.6f}",
           'std_accuracy': f"{metrics['std_accuracy']:.6f}",
           'minimum_fold': f"{metrics['min_fold']:.6f}",
           'macro_f1': f"{metrics['macro_f1']:.6f}",
           'balanced_accuracy': f"{metrics['balanced_accuracy']:.6f}",
           'train_score': f"{metrics['train_score']:.6f}",
           'overfit_gap': f"{metrics['overfit_gap']:.6f}",
           'runtime': f"{metrics['runtime']:.2f}",
           'accepted': accepted, 'notes': notes[:200]}
    pd.DataFrame([row]).to_csv(p, mode='a', header=not p.exists(), index=False)

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
# FEATURE ENGINEERING
# ================================================================
def build_mega_features(train, test):
    """Aggressive feature engineering."""
    t, te = train.copy(), test.copy()

    # Base engineered
    X = get_all_features(t)
    Xte = get_all_features(te)

    # Signal processing features for weekly scores
    week = t[WEEK_COLS].values
    week_te = te[WEEK_COLS].values

    # Rolling correlations with time
    for i in range(1, 13):
        X[f'week_lag_{i}'] = np.roll(week[:, 0], i, axis=0) if i > 0 else week[:, 0]
    # Don't do lags for test (no shift needed)

    # Week-to-week changes (acceleration)
    w_diff = np.diff(week, axis=1)
    X['week_acceleration'] = np.mean(np.diff(w_diff, axis=1), axis=1)

    # Activity-to-week alignment
    act = t[ACTIVITY_COLS].values; act_te = te[ACTIVITY_COLS].values

    # Best 3 weeks
    X['week_top3_mean'] = np.sort(week, axis=1)[:, -3:].mean(axis=1)
    X['week_bot3_mean'] = np.sort(week, axis=1)[:, :3].mean(axis=1)
    X['week_top_bot_ratio'] = X['week_top3_mean'] / (np.abs(X['week_bot3_mean']) + 0.01)

    # Activity time-of-day grouping (early/late split by even/odd days)
    act_even = act[:, ::2]; act_odd = act[:, 1::2]
    X['act_even_mean'] = act_even.mean(axis=1); X['act_odd_mean'] = act_odd.mean(axis=1)
    X['act_even_odd_ratio'] = X['act_even_mean'] / (X['act_odd_mean'] + 0.1)

    # Task + activity interaction
    X['task_act_product'] = t['tugas_selesai'].values * X['activity_mean'] / 100
    X['task_activity_efficiency'] = X['task_completion_ratio'] / (X['activity_std'] + 0.01)

    # Tryout deep interactions
    X['tryout_rank'] = t['skor_tryout'].rank(pct=True)
    X['tryout_x_week'] = t['skor_tryout'] * X['week_mean']
    X['tryout_x_task'] = t['skor_tryout'] * X['task_completion_ratio']

    # Behavioral composites
    X['behav_pos'] = (t['skor_motivasi'] > 0).astype(int) + (t['skor_kedisiplinan'] > 0).astype(int) + \
                     (t['skor_ekstrakurikuler'] > 0).astype(int) + (t['indeks_kehadiran'] > 0).astype(int)
    X['behav_sum'] = t['skor_motivasi'] + t['skor_kedisiplinan'] + t['skor_ekstrakurikuler'] + t['indeks_kehadiran'] + t['skor_literasi']
    X['behav_abs_sum'] = np.abs(t['skor_motivasi']) + np.abs(t['skor_kedisiplinan']) + np.abs(t['skor_ekstrakurikuler'])

    # PCA at different components
    for nc in [3, 5, 8, 10]:
        pca = PCA(n_components=nc, random_state=SEED)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(t[WEEK_COLS]))
        Xte_pca = pca.transform(StandardScaler().fit_transform(te[WEEK_COLS]))
        for i in range(nc):
            X[f'pca_w{nc}_{i}'] = X_pca[:, i]
            Xte[f'pca_w{nc}_{i}'] = Xte_pca[:, i]

    # ICA
    ica = FastICA(n_components=5, random_state=SEED)
    X_ica = ica.fit_transform(StandardScaler().fit_transform(t[WEEK_COLS]))
    Xte_ica = ica.transform(StandardScaler().fit_transform(te[WEEK_COLS]))
    for i in range(5):
        X[f'ica_w_{i}'] = X_ica[:, i]; Xte[f'ica_w_{i}'] = Xte_ica[:, i]

    # Bin features
    X['tugas_selesai_bin'] = pd.qcut(t['tugas_selesai'].values, 4, labels=False, duplicates='drop')
    X['skor_tryout_bin'] = pd.qcut(t['skor_tryout'].values, 4, labels=False, duplicates='drop')
    X['week_mean_bin'] = pd.qcut(X['week_mean'].values, 4, labels=False, duplicates='drop')

    # Interactions between top features
    top = ['tugas_selesai', 'skor_tryout', 'week_mean', 'tugas_diberikan',
           'activity_mean', 'task_completion_ratio', 'skor_motivasi', 'skor_kedisiplinan']
    avail = [c for c in top if c in X.columns]
    for i in range(len(avail)):
        for j in range(i+1, len(avail)):
            a, b = avail[i], avail[j]
            X[f'{a}_x_{b}'] = X[a] * X[b]
            Xte[f'{a}_x_{b}'] = Xte[a] * Xte[b]
            if a != b:
                X[f'{a}_div_{b}'] = X[a] / (X[b] + 0.01)
                Xte[f'{a}_div_{b}'] = Xte[a] / (Xte[b] + 0.01)

    # Clip
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        s_min, s_max = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(s_min, s_max)
    for col in Xte.select_dtypes(include=[np.number]).columns:
        Xte[col] = Xte[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure column alignment
    common = [c for c in X.columns if c in Xte.columns]
    X, Xte = X[common], Xte[common]

    return X.fillna(0).values, Xte.fillna(0).values


# ================================================================
# MULTI-MODEL STACKING
# ================================================================
def train_base_models(X, y, X_test, seed=SEED):
    """Train diverse base models and collect OOF + test predictions."""
    cv = get_cv(seed)
    n = len(y)

    model_dict = {}
    models_to_try = [
        ('SVC_C50', SVC(C=50, gamma='auto', probability=True, random_state=seed)),
        ('SVC_C10', SVC(C=10, gamma='scale', probability=True, random_state=seed)),
        ('RF500', RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        ('ET500', ExtraTreesClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        ('GB300', GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, min_samples_leaf=5, random_state=seed)),
        ('HGB', HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, random_state=seed)),
        ('LR_C1', LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=seed, n_jobs=-1)),
        ('LR_01', LogisticRegression(C=0.1, solver='lbfgs', max_iter=2000, random_state=seed, n_jobs=-1)),
        ('Ridge', RidgeClassifier(alpha=1.0, random_state=seed)),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN15', KNeighborsClassifier(n_neighbors=15, weights='distance')),
        ('KNN31', KNeighborsClassifier(n_neighbors=31, weights='distance')),
    ]

    try:
        import xgboost as xgb
        models_to_try.append(('XGB', xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0)))
    except: pass
    try:
        import lightgbm as lgb
        models_to_try.append(('LGB', lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)))
    except: pass
    try:
        import catboost as cb
        models_to_try.append(('CB', cb.CatBoostClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_seed=seed, verbose=0)))
    except: pass

    mnames, moofs, mprobs, mtest = [], {}, {}, {}

    for mname, model in models_to_try:
        try:
            r = evaluate(X, y, model, mname, X_test=X_test)
            mnames.append(mname)
            moofs[mname] = r['oof_predictions']
            mprobs[mname] = r['oof_probabilities']
            mtest[mname] = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r.get('test_preds',
                np.zeros((10, X_test.shape[0]))))

            eid = f"EXP-BASE-{mname}"
            log_exp(eid, "PHASE4", f"Base model {mname}", "mega_features", mname,
                    str(model.get_params())[:150], seed, r,
                    "Ya" if r['oof_accuracy'] > 0.55 else "Tidak", "")
        except Exception as e:
            print(f"  {mname:30s} FAILED: {e}")

    # Build stacked features: all probs concatenated
    stack_cols = []
    for m in mnames:
        stack_cols.append(mprobs[m])
    X_stack = np.column_stack(stack_cols)

    # Also add OOF preds as features
    pred_cols = []
    for m in mnames:
        pred_cols.append(moofs[m].reshape(-1, 1))
    X_stack = np.column_stack([X_stack] + pred_cols)

    print(f"\n  Stack features: {X_stack.shape} from {len(mnames)} models")

    # Build test stack
    test_stack_cols = []
    for m in mnames:
        # Re-train on full data to get test probs
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_te_s = scaler.transform(X_test)
        model_clone = [md for mn, md in models_to_try if mn == m][0]
        model_clone = copy.deepcopy(model_clone)

        try:
            model_clone.fit(X_s, y)
            if hasattr(model_clone, 'predict_proba'):
                test_probs = model_clone.predict_proba(X_te_s)
            else:
                test_probs = np.zeros((X_test.shape[0], 4))
            test_preds = np.ravel(model_clone.predict(X_te_s))
        except:
            test_probs = np.zeros((X_test.shape[0], 4))
            test_preds = np.zeros(X_test.shape[0])

        test_stack_cols.append(test_probs)

    X_test_stack = np.column_stack(test_stack_cols + [np.zeros((X_test.shape[0], len(mnames)))])

    return mnames, moofs, mprobs, X_stack, X_test_stack


def meta_tune(X_stack, y, X_test_stack, sample_sub):
    """Tune meta-models on stack features."""
    best_m = None
    best_r = None

    metas = [
        ('LR_C1', LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        ('LR_C01', LogisticRegression(C=0.1, max_iter=2000, random_state=SEED)),
        ('LR_C10', LogisticRegression(C=10.0, max_iter=2000, random_state=SEED)),
        ('LR_l2', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=SEED)),
        ('RidgeCV', RidgeClassifier(alpha=1.0, random_state=SEED)),
        ('Ridge_01', RidgeClassifier(alpha=0.1, random_state=SEED)),
        ('GB_meta', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED)),
        ('RF_meta', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED, n_jobs=-1)),
        ('HGB_meta', HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=SEED)),
        ('SVC_meta', SVC(C=10, gamma='scale', probability=True, random_state=SEED)),
    ]

    for mname, meta in metas:
        try:
            r = evaluate(X_stack, y, meta, f"Meta_{mname}", X_test=X_test_stack)
            eid = f"EXP-META-{mname}"
            log_exp(eid, "PHASE4-STACK", f"Meta {mname}", "stack_probs", f"Meta_{mname}",
                    str(meta.get_params())[:150], SEED, r,
                    "Ya" if (best_m is None or r['oof_accuracy'] > best_m['oof_accuracy'] + 0.001) else "Tidak", "")
            if best_m is None or r['oof_accuracy'] > best_m['oof_accuracy']:
                best_m = mname
                best_r = r
        except Exception as e:
            print(f"  Meta_{mname:25s} FAILED: {e}")

    if best_r is not None:
        make_sub(best_r['oof_predictions'], sample_sub, "submission_meta_best.csv")
    return best_m, best_r


# ================================================================
# MAIN
# ================================================================
best = {'accuracy': 0.5869, 'model': 'Stack_LR', 'exp_id': 'EXP-027'}
no_improve = 0
experiment_count = 28

def main():
    global best, no_improve, experiment_count

    train, test, sample_sub = load_data()
    y = train['target'].values

    print(f"\n{'='*60}")
    print(f"PHASE 4: AGGRESSIVE STACKING + DEEP FEATURES")
    print(f"Starting from: {best['accuracy']:.4f} ({best['model']})")
    print(f"Remaining to 0.70: {0.70 - best['accuracy']:.4f}")
    print(f"{'='*60}")

    # Build mega features
    print("\n--- Building mega feature set ---")
    X, X_test = build_mega_features(train, test)
    print(f"  Features: {X.shape[1]}")

    # Train diverse base models
    print("\n--- Training base model zoo ---")
    mnames, moofs, mprobs, X_stack, X_test_stack = train_base_models(X, y, X_test)

    # Meta-model tuning
    print("\n--- Meta-model tuning ---")
    best_meta, best_meta_r = meta_tune(X_stack, y, X_test_stack, sample_sub)

    if best_meta_r and best_meta_r['oof_accuracy'] > best['accuracy']:
        best.update(accuracy=best_meta_r['oof_accuracy'], model=f"Meta_{best_meta}",
                     macro_f1=best_meta_r['macro_f1'])
        checkpoint("PHASE4-BEST", f"Meta_{best_meta}", best_meta_r, "mega+stack", "")
        no_improve = 0

    # TWO-LEVEL STACKING
    print(f"\n{'='*60}")
    print("TWO-LEVEL STACKING")
    print(f"{'='*60}")

    cv = get_cv()
    n = len(y)
    X_stack2 = np.zeros((n, len(mnames) * 4))
    X_test_stack2 = np.zeros((X_test.shape[0], len(mnames) * 4))

    for fi, (tr, val) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[tr], X[val]
        y_tr = y[tr]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_test)

        col_idx = 0
        for mname in mnames:
            # Get model class
            try:
                if 'SVC' in mname:
                    m_class = SVC(C=50 if 'C50' in mname else 10, gamma='auto' if 'C50' in mname else 'scale',
                                  probability=True, random_state=SEED+fi)
                elif 'RF' in mname:
                    m_class = RandomForestClassifier(n_estimators=500, max_depth=14, random_state=SEED+fi, n_jobs=-1)
                elif 'ET' in mname:
                    m_class = ExtraTreesClassifier(n_estimators=500, max_depth=12, random_state=SEED+fi, n_jobs=-1)
                elif 'GB' in mname:
                    m_class = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED+fi)
                elif 'HGB' in mname:
                    m_class = HistGradientBoostingClassifier(max_iter=300, max_depth=5, random_state=SEED+fi)
                elif 'LR_C1' == mname:
                    m_class = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED+fi)
                elif 'LR_01' == mname:
                    m_class = LogisticRegression(C=0.1, max_iter=2000, random_state=SEED+fi)
                elif 'Ridge' in mname:
                    m_class = RidgeClassifier(alpha=1.0, random_state=SEED+fi)
                elif 'LDA' in mname:
                    m_class = LinearDiscriminantAnalysis()
                elif 'KNN' in mname:
                    k = int(mname.replace('KNN',''))
                    m_class = KNeighborsClassifier(n_neighbors=k, weights='distance')
                elif 'XGB' in mname:
                    import xgboost as xgb
                    m_class = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED+fi, n_jobs=-1, verbosity=0)
                elif 'LGB' in mname:
                    import lightgbm as lgb
                    m_class = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=SEED+fi, verbose=-1)
                elif 'CB' in mname:
                    import catboost as cb
                    m_class = cb.CatBoostClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_seed=SEED+fi, verbose=0)
                else:
                    continue

                m_class.fit(X_tr_s, y_tr)
                val_prob = m_class.predict_proba(X_val_s)
                te_prob = m_class.predict_proba(X_te_s)
                X_stack2[val, col_idx:col_idx+4] = val_prob
                X_test_stack2[:, col_idx:col_idx+4] += te_prob / 10  # average across folds
            except:
                X_stack2[val, col_idx:col_idx+4] = np.ones((len(val), 4)) * 0.25
            col_idx += 4

    print(f"\n  2-level stack features: {X_stack2.shape}")

    # Meta on level 2
    for mname, meta in [
        ('LR_C1_l2', LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        ('LR_01_l2', LogisticRegression(C=0.1, max_iter=2000, random_state=SEED)),
        ('Ridge_l2', RidgeClassifier(alpha=1.0, random_state=SEED)),
    ]:
        experiment_count += 1
        eid = f"EXP-{experiment_count:03d}"
        r = evaluate(X_stack2, y, meta, f"Meta2_{mname}", X_test=X_test_stack2)
        log_exp(eid, "PHASE4-L2", f"Level-2 meta {mname}", "stack_l2", f"Meta2_{mname}",
                str(meta.get_params())[:150], SEED, r,
                "Ya" if r['oof_accuracy'] > best['accuracy']+0.002 else "Tidak", "")
        if r['oof_accuracy'] > best['accuracy'] + 0.002:
            best.update(accuracy=r['oof_accuracy'], model=f"Meta2_{mname}", macro_f1=r['macro_f1'])
            checkpoint(eid, f"Meta2_{mname}", r, "stack_l2", "")
            no_improve = 0
        else:
            no_improve += 1
        if 'test_preds' in r:
            ts = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, r['test_preds'])
            make_sub(ts, sample_sub, f"submission_{eid}.csv")

    # ================================================================
    # FINAL
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 4 COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {experiment_count}")
    print(f"Best: {best['model']} = {best['accuracy']:.4f}")

    with open(EXP_DIR / "reports" / "phase4_summary.json", 'w') as f:
        json.dump(best, f, indent=2)

    remaining = 0.70 - best['accuracy']
    if best['accuracy'] >= 0.70:
        print("\n*** TARGET 0.70 ACHIEVED! ***")
    else:
        print(f"Remaining to 0.70: {remaining:.4f}")
        if remaining < 0.015:
            print("Very close! Threshold optimization + seed averaging will bridge it.")
        elif remaining < 0.05:
            print("Need more: try pseudo-labeling style approaches.")
        else:
            print("Further improvement needed: deeper exploration required.")
        print(f"\nNext: Phase 5 — Seed averaging, threshold opt, pseudo-labeling")

if __name__ == "__main__":
    main()
