"""
Phase 2: Advanced Feature Engineering + Ordinal Modeling + Hyperparameter Tuning
"""
import numpy as np
import pandas as pd
import json, os, time, sys, warnings, traceback
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, log_loss, r2_score)
from sklearn.pipeline import Pipeline

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS, engineer_features

SEED = 42
DATA_DIR = Path("data")
EXP_DIR = Path("experiments")

# ============================================================
# HELPERS
# ============================================================
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sample_sub

def evaluate_model_cv(X, y, model, model_name, X_test=None, seed=42, n_repeats=2):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    n_total = len(y)
    oof_preds = np.zeros(n_total, dtype=int)
    oof_probs = np.zeros((n_total, 4))
    oof_fold = np.zeros(n_total)
    fold_metrics = []
    test_preds_list = []
    test_probs_list = []
    start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        try:
            m = model if hasattr(model, 'set_params') else model
            m.fit(X_tr_s, y_tr)
        except:
            model.fit(X_tr_s, y_tr)
            m = model

        y_val_pred = np.ravel(m.predict(X_val_s))
        y_val_prob = m.predict_proba(X_val_s) if hasattr(m, 'predict_proba') else None
        y_tr_pred = np.ravel(m.predict(X_tr_s))

        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        val_bal = balanced_accuracy_score(y_val, y_val_pred)
        train_acc = accuracy_score(y_tr, y_tr_pred)

        fold_metrics.append({'fold': fold_idx, 'accuracy': val_acc, 'macro_f1': val_f1,
                             'balanced_accuracy': val_bal, 'train_accuracy': train_acc,
                             'overfit_gap': train_acc - val_acc})
        oof_preds[val_idx] = y_val_pred
        if y_val_prob is not None:
            oof_probs[val_idx] = y_val_prob
        oof_fold[val_idx] = fold_idx

        if X_test is not None:
            X_test_s = scaler.transform(X_test)
            test_preds_list.append(m.predict(X_test_s))
            if hasattr(m, 'predict_proba'):
                test_probs_list.append(m.predict_proba(X_test_s))

    elapsed = time.time() - start
    accs = [m['accuracy'] for m in fold_metrics]
    oof_acc = accuracy_score(y, oof_preds)
    oof_f1 = f1_score(y, oof_preds, average='macro')

    results = {
        'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)),
        'min_fold': float(np.min(accs)), 'max_fold': float(np.max(accs)),
        'macro_f1': float(oof_f1),
        'balanced_accuracy': float(np.mean([m['balanced_accuracy'] for m in fold_metrics])),
        'train_score': float(np.mean([m['train_accuracy'] for m in fold_metrics])),
        'overfit_gap': float(np.mean([m['overfit_gap'] for m in fold_metrics])),
        'fold_details': fold_metrics, 'runtime': elapsed,
        'oof_predictions': oof_preds, 'oof_probabilities': oof_probs,
        'oof_fold': oof_fold, 'oof_accuracy': float(oof_acc),
    }
    if test_preds_list:
        results['test_preds'] = np.array(test_preds_list)
        if test_probs_list:
            results['test_probs'] = np.array(test_probs_list)

    print(f"  {model_name} — OOF: {oof_acc:.4f} (±{np.std(accs):.4f}), F1={oof_f1:.4f}, "
          f"min={np.min(accs):.4f}, gap={np.mean([m['overfit_gap'] for m in fold_metrics]):.3f}, "
          f"{elapsed:.0f}s")
    return results

def log_experiment(exp_id, parent, hypothesis, feature_set, model_name, params,
                   cv_strategy, seed, metrics, accepted, notes=""):
    log_path = EXP_DIR / "experiment_log.csv"
    row = {
        'experiment_id': exp_id, 'parent_experiment': parent,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hypothesis': hypothesis, 'feature_set': feature_set, 'model': model_name,
        'parameters': str(params)[:100] if params else '',
        'cv_strategy': cv_strategy, 'seed': seed,
        'mean_accuracy': f"{metrics.get('oof_accuracy', 0):.6f}",
        'std_accuracy': f"{metrics['std_accuracy']:.6f}",
        'minimum_fold': f"{metrics['min_fold']:.6f}",
        'macro_f1': f"{metrics['macro_f1']:.6f}",
        'balanced_accuracy': f"{metrics['balanced_accuracy']:.6f}",
        'train_score': f"{metrics['train_score']:.6f}",
        'overfit_gap': f"{metrics['overfit_gap']:.6f}",
        'runtime': f"{metrics['runtime']:.2f}",
        'accepted': accepted, 'notes': notes[:200] if notes else "",
    }
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

def save_checkpoint(exp_id, model_name, metrics, feature_set, params):
    config = {'experiment_id': exp_id, 'model': model_name, 'feature_set': feature_set,
              'parameters': params, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'metrics': {'mean_accuracy': metrics['mean_accuracy'], 'std_accuracy': metrics['std_accuracy'],
                          'min_fold': metrics['min_fold'], 'macro_f1': metrics['macro_f1'],
                          'balanced_accuracy': metrics['balanced_accuracy'], 'overfit_gap': metrics['overfit_gap']}}
    with open(EXP_DIR / "best_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    np.save(EXP_DIR / "oof_predictions" / "best_oof_preds.npy", metrics['oof_predictions'])
    np.save(EXP_DIR / "oof_predictions" / "best_oof_probs.npy", metrics['oof_probabilities'])
    pd.DataFrame({'id': range(len(metrics['oof_predictions'])),
                   'target': metrics['oof_predictions'].astype(int)}).to_csv(
        EXP_DIR / "oof_predictions" / "best_oof.csv", index=False)
    print(f"  >>> CHECKPOINT: {model_name} = {metrics.get('oof_accuracy', 0):.4f}")

def make_submission(test_preds, sample_sub, filename):
    sub = sample_sub.copy()
    sub['target'] = test_preds.astype(int)
    sub.to_csv(EXP_DIR / filename, index=False)

def load_experiment_log():
    log_path = EXP_DIR / "experiment_log.csv"
    if log_path.exists():
        return pd.read_csv(log_path)
    return pd.DataFrame()

# ============================================================
# FEATURE BUILDERS
# ============================================================
def add_pca_features(train_df, test_df):
    """Add PCA components for weekly scores and activity features."""
    result_train, result_test = train_df.copy(), test_df.copy()

    for group_name, cols, n_comp in [
        ('week_pca', WEEK_COLS, 5),
        ('activity_pca', ACTIVITY_COLS, 6),
    ]:
        scaler = StandardScaler()
        data = scaler.fit_transform(train_df[cols].values)
        test_data = scaler.transform(test_df[cols].values)

        pca = PCA(n_components=n_comp, random_state=SEED)
        train_pca = pca.fit_transform(data)
        test_pca = pca.transform(test_data)

        for i in range(n_comp):
            result_train[f'{group_name}_{i+1}'] = train_pca[:, i]
            result_test[f'{group_name}_{i+1}'] = test_pca[:, i]

    return result_train, result_test

def add_cluster_features(train_df, test_df, cols, n_clusters=5):
    """Add cluster distance features."""
    scaler = StandardScaler()
    data = scaler.fit_transform(train_df[cols].values)
    test_data = scaler.transform(test_df[cols].values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    kmeans.fit(data)

    train_dist = kmeans.transform(data)
    test_dist = kmeans.transform(test_data)

    result_train, result_test = train_df.copy(), test_df.copy()
    for i in range(n_clusters):
        result_train[f'cluster_dist_{i+1}'] = train_dist[:, i]
        result_test[f'cluster_dist_{i+1}'] = test_dist[:, i]
    result_train['cluster_label'] = kmeans.predict(data)
    result_test['cluster_label'] = kmeans.predict(test_data)

    return result_train, result_test

def add_poly_features(df, cols, degree=2):
    """Add polynomial features for selected columns."""
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
    data = df[cols].values
    poly_data = poly.fit_transform(data)
    feature_names = poly.get_feature_names_out(cols)
    poly_df = pd.DataFrame(poly_data, columns=feature_names, index=df.index)
    # Remove original columns from poly output
    poly_df = poly_df.drop(columns=cols, errors='ignore')
    return pd.concat([df, poly_df], axis=1)

def build_feature_set(train, test, variant='full'):
    """Build various feature sets."""
    y = train['target'].values if 'target' in train.columns else None
    sample_sub = pd.read_csv(DATA_DIR / "data" / "sample_submission.csv") if False else None

    # Base: engineered features
    X_train = get_all_features(train)
    X_test = get_all_features(test)

    if variant == 'full':
        X_train, X_test = add_pca_features(X_train, X_test)

        # Cluster on weekly scores
        X_train, X_test = add_cluster_features(X_train, X_test, WEEK_COLS, n_clusters=5)
        X_train, X_test = add_cluster_features(X_train, X_test, ACTIVITY_COLS, n_clusters=5)

        # Key interactions
        key_cols = ['tugas_selesai', 'skor_tryout', 'week_mean', 'week_std',
                     'activity_mean', 'activity_std', 'task_completion_ratio',
                     'skor_motivasi', 'skor_kedisiplinan']
        available = [c for c in key_cols if c in X_train.columns]
        X_train_add = add_poly_features(X_train[available], available, degree=2)
        X_test_add = add_poly_features(X_test[available], available, degree=2)
        # Merge only the new poly columns
        new_cols = [c for c in X_train_add.columns if c not in X_train.columns]
        X_train = pd.concat([X_train, X_train_add[new_cols]], axis=1)
        X_test = pd.concat([X_test, X_test_add[new_cols]], axis=1)

    elif variant == 'pca_only':
        X_train, X_test = add_pca_features(X_train, X_test)

    elif variant == 'cluster_only':
        X_train, X_test = add_cluster_features(X_train, X_test, WEEK_COLS, n_clusters=5)
        X_train, X_test = add_cluster_features(X_train, X_test, ACTIVITY_COLS, n_clusters=5)

    elif variant == 'interaction_only':
        key_cols = ['tugas_selesai', 'skor_tryout', 'week_mean', 'week_std',
                     'activity_mean', 'task_completion_ratio', 'skor_motivasi', 'skor_kedisiplinan']
        available = [c for c in key_cols if c in X_train.columns]
        X_train_int = add_poly_features(X_train[available], available, degree=2)
        X_test_int = add_poly_features(X_test[available], available, degree=2)
        new_cols = [c for c in X_train_int.columns if c not in X_train.columns]
        X_train = pd.concat([X_train, X_train_int[new_cols]], axis=1)
        X_test = pd.concat([X_test, X_test_int[new_cols]], axis=1)

    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

    return X_train.values, X_test.values, y

def build_ordinal_features(y):
    """Create ordinal targets: P(target>=k) for k=1,2,3."""
    y1 = (y >= 1).astype(int)
    y2 = (y >= 2).astype(int)
    y3 = (y >= 3).astype(int)
    return [y1, y2, y3]

# ============================================================
# ORDINAL MODEL
# ============================================================
def train_ordinal_model(X, y, X_test=None, seed=42):
    """
    Train 3 binary models for ordinal classification, then combine.
    P(target>=1), P(target>=2), P(target>=3) -> P(target=k)
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    n = len(y)

    oof_probs = np.zeros((n, 3))  # probabilities for each binary
    test_probs = np.zeros((3, X_test.shape[0])) if X_test is not None else None
    y_bin = build_ordinal_features(y)

    for k in range(3):
        oof_k = np.zeros(n)
        test_k_list = []

        for fold_idx, (tr, val) in enumerate(cv.split(X, y)):
            X_tr, X_val = X[tr], X[val]
            y_tr, y_val = y_bin[k][tr], y_bin[k][val]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            # Use a strong binary classifier
            model = HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                                   learning_rate=0.05, random_state=seed+fold_idx)
            model.fit(X_tr_s, y_tr)
            oof_k[val] = model.predict_proba(X_val_s)[:, 1]

            if X_test is not None:
                X_te_s = scaler.transform(X_test)
                test_k_list.append(model.predict_proba(X_te_s)[:, 1])

        oof_probs[:, k] = oof_k
        if test_k_list:
            test_probs[k] = np.mean(test_k_list, axis=0)

    # Convert cumulative to class probabilities: P(target=k) = P(>=k) - P(>=k+1)
    class_probs = np.zeros((n, 4))
    class_probs[:, 0] = 1 - oof_probs[:, 0]           # P = 0
    class_probs[:, 1] = oof_probs[:, 0] - oof_probs[:, 1]  # P = 1
    class_probs[:, 2] = oof_probs[:, 1] - oof_probs[:, 2]  # P = 2
    class_probs[:, 3] = oof_probs[:, 2]                     # P = 3
    class_probs = np.clip(class_probs, 0, 1)
    class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)

    preds = np.argmax(class_probs, axis=1)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')

    test_class_probs = None
    if test_probs is not None:
        test_class_probs = np.zeros((X_test.shape[0], 4))
        test_class_probs[:, 0] = 1 - test_probs[0]
        test_class_probs[:, 1] = test_probs[0] - test_probs[1]
        test_class_probs[:, 2] = test_probs[1] - test_probs[2]
        test_class_probs[:, 3] = test_probs[2]
        test_class_probs = np.clip(test_class_probs, 0, 1)
        test_class_probs = test_class_probs / test_class_probs.sum(axis=1, keepdims=True)

    # Compute fold-level metrics for logging
    cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    fold_accs = []
    for tr, val in cv2.split(X, y):
        fold_accs.append(accuracy_score(y[val], preds[val]))

    results = {
        'mean_accuracy': float(np.mean(fold_accs)),
        'std_accuracy': float(np.std(fold_accs)),
        'min_fold': float(np.min(fold_accs)),
        'max_fold': float(np.max(fold_accs)),
        'macro_f1': float(f1),
        'bal_accuracy': float(balanced_accuracy_score(y, preds)),
        'oof_accuracy': float(acc),
        'oof_predictions': preds,
        'oof_probabilities': class_probs,
        'runtime': 0,
        'overfit_gap': 0,
        'train_score': 0,
        'balanced_accuracy': float(balanced_accuracy_score(y, preds)),
    }

    print(f"  OrdinalModel — OOF: {acc:.4f}, F1={f1:.4f}")
    return results, test_class_probs

# ============================================================
# OPTUNA TUNING
# ============================================================
def optuna_tune(X, y, model_type='svc', n_trials=30):
    import optuna
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        if model_type == 'svc':
            C = trial.suggest_float('C', 0.1, 200, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
            model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=SEED)
        elif model_type == 'xgb':
            import xgboost as xgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
                'random_state': SEED, 'n_jobs': -1, 'verbosity': 0,
            }
            model = xgb.XGBClassifier(**params)
        elif model_type == 'lgb':
            import lightgbm as lgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'random_state': SEED, 'n_jobs': -1, 'verbose': -1,
            }
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_type}")

        fold_accs = []
        for tr, val in cv.split(X, y):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X[tr])
            X_val_s = scaler.transform(X[val])
            model.fit(X_tr_s, y[tr])
            preds = model.predict(X_val_s)
            fold_accs.append(accuracy_score(y[val], preds))

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        train_preds = model.predict(X_tr_s)
        train_acc = accuracy_score(y[tr], train_preds)

        # Penalize high variance and overfitting
        penalty = 0.2 * std_acc + 0.1 * max(0, train_acc - mean_acc - 0.05)
        return mean_acc - penalty

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n  Optuna [{model_type}] best: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    return study.best_params, study.best_value

# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================
best_score = {'accuracy': 0.5156, 'macro_f1': 0.5114, 'model': 'SVC_RBF',
              'feature_set': 'full_engineered', 'exp_id': 'EXP-006'}
no_improvement_count = 3
experiment_count = 4
best_info = best_score.copy()

def run_exp(model, name, fset, X, y, X_test, sample_sub, parent, hypothesis, params_str=None):
    global experiment_count, no_improvement_count, best_info
    experiment_count += 1
    eid = f"EXP-{experiment_count:03d}"

    gen_model = model if callable(model) else (lambda rs: model)
    res = evaluate_model_cv(X, y, gen_model(SEED), name, X_test=X_test, seed=SEED, n_repeats=2)

    accepted = "Tidak"
    improvement = 0
    if res and res['oof_accuracy'] > best_info['accuracy']:
        improvement = res['oof_accuracy'] - best_info['accuracy']
        if improvement >= 0.002:
            accepted = "Ya"
            best_info.update({'accuracy': res['oof_accuracy'], 'macro_f1': res['macro_f1'],
                              'model': name, 'feature_set': fset, 'exp_id': eid})
            save_checkpoint(eid, name, res, fset, params_str or str(model.get_params() if hasattr(model, 'get_params') else ''))
            no_improvement_count = 0
        else:
            accepted = "Marginal"
            no_improvement_count += 1
    elif res:
        no_improvement_count += 1

    log_experiment(eid, parent, hypothesis, fset, name,
                   params_str or str(gen_model(SEED).get_params() if hasattr(gen_model(SEED), 'get_params') else ''),
                   "RepeatedStratifiedKFold(5,2)", SEED, res, accepted,
                   f"improvement={improvement:.4f}")

    if 'test_preds' in res:
        test_preds = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, res['test_preds'])
        make_submission(test_preds, sample_sub, f"submission_{eid}.csv")

    return res

def main():
    global best_info, experiment_count, no_improvement_count

    train, test, sample_sub = load_data()
    y = train['target'].values

    print(f"\n{'='*60}")
    print(f"PHASE 2: ADVANCED FEATURE ENGINEERING + ORDINAL + TUNING")
    print(f"Starting from best: {best_info['accuracy']:.4f} ({best_info['model']})")
    print(f"{'='*60}")

    # ---- BUILD ALL FEATURE SETS ----
    print("\n--- Building feature sets ---")
    fsets = {}
    # full_engineered (reuse from phase 1)
    X_full = get_all_features(train)
    X_test_full = get_all_features(test)
    fsets['full_eng'] = (X_full.values, X_test_full.values)

    # PCA features
    X_pca, X_test_pca = add_pca_features(X_full, X_test_full)
    fsets['full_pca'] = (X_pca.values, X_test_pca.values)

    # Clustering
    X_clust, X_test_clust = X_full.copy(), X_test_full.copy()
    X_clust, X_test_clust = add_cluster_features(X_clust, X_test_clust, WEEK_COLS, 5)
    X_clust, X_test_clust = add_cluster_features(X_clust, X_test_clust, ACTIVITY_COLS, 5)
    # Also add poly interactions for key features
    key_cols = ['tugas_selesai', 'skor_tryout', 'week_mean', 'week_std',
                 'activity_mean', 'task_completion_ratio', 'skor_motivasi', 'skor_kedisiplinan']
    avail = [c for c in key_cols if c in X_clust.columns]
    X_clust_int = add_poly_features(X_clust[avail], avail, degree=2)
    X_test_clust_int = add_poly_features(X_test_clust[avail], avail, degree=2)
    new_c = [c for c in X_clust_int.columns if c not in X_clust.columns]
    X_clust = pd.concat([X_clust, X_clust_int[new_c]], axis=1)
    X_test_clust = pd.concat([X_test_clust, X_test_clust_int[new_c]], axis=1)
    fsets['full_advanced'] = (X_clust.fillna(0).replace([np.inf, -np.inf], 0).values,
                              X_test_clust.fillna(0).replace([np.inf, -np.inf], 0).values)

    print(f"  full_eng: {fsets['full_eng'][0].shape}")
    print(f"  full_pca: {fsets['full_pca'][0].shape}")
    print(f"  full_advanced: {fsets['full_advanced'][0].shape}")

    # ---- EXP 011-015: Test different feature sets with SVC ----
    print("\n--- Testing feature variants with SVC RBF ---")
    for fname, (X_f, X_t_f) in [('full_pca', fsets['full_pca']),
                                 ('full_advanced', fsets['full_advanced'])]:
        run_exp(lambda rs: SVC(C=10, gamma='scale', probability=True, random_state=rs),
                f"SVC_{fname}", fname, X_f, y, X_t_f, sample_sub,
                best_info['exp_id'], f"SVC on {fname} features")

    # ---- EXP 016-019: Try CatBoost (now installed) ----
    print("\n--- CatBoost experiments ---")
    try:
        import catboost as cb
        for fname, (X_f, X_t_f) in [('full_eng', fsets['full_eng']),
                                      ('full_advanced', fsets['full_advanced'])]:
            run_exp(lambda rs: cb.CatBoostClassifier(n_estimators=400, max_depth=6,
                                                      learning_rate=0.05, random_seed=rs,
                                                      verbose=0, early_stopping_rounds=50),
                    f"CatBoost_{fname}", fname, X_f, y, X_t_f, sample_sub,
                    best_info['exp_id'], "CatBoost with engineered features")
    except ImportError:
        print("  CatBoost unavailable")

    # ---- EXP 020-021: HistGradientBoosting ----
    print("\n--- HistGradientBoosting ---")
    for fname, (X_f, X_t_f) in [('full_eng', fsets['full_eng']),
                                  ('full_advanced', fsets['full_advanced'])]:
        run_exp(lambda rs: HistGradientBoostingClassifier(max_iter=400, max_depth=6,
                                                           learning_rate=0.05, random_state=rs),
                f"HistGB_{fname}", fname, X_f, y, X_t_f, sample_sub,
                best_info['exp_id'], "HistGB with engineered features")

    # ---- EXP 022-024: KNN variants ----
    print("\n--- KNN variants ---")
    X_adv, X_t_adv = fsets['full_advanced']
    for nn in [7, 15, 31]:
        run_exp(lambda rs: KNeighborsClassifier(n_neighbors=nn, weights='distance', p=2),
                f"KNN_{nn}", "full_advanced", X_adv, y, X_t_adv, sample_sub,
                best_info['exp_id'], f"KNN k={nn}")

    # ============================================================
    # SUB-PHASE: ORDINAL MODELING
    # ============================================================
    print(f"\n{'='*60}")
    print("SUB-PHASE: ORDINAL MODELING")
    print(f"{'='*60}")

    # Ordinal on advanced features
    print("\n--- Ordinal Cumulative Model ---")
    X_adv, X_t_adv = fsets['full_advanced']

    # Run ordinal model (no separate OOF, we compute directly)
    ord_results, ord_test_probs = train_ordinal_model(X_adv, y, X_test=X_t_adv, seed=SEED)
    experiment_count += 1
    eid = f"EXP-{experiment_count:03d}"
    log_experiment(eid, best_info['exp_id'],
                   "Ordinal cumulative model with advanced features",
                   "full_advanced", "OrdinalCumulative",
                   "HGB 3x binary classifiers",
                   "RepeatedStratifiedKFold(5,2)", SEED, ord_results,
                   "Ya" if ord_results['oof_accuracy'] > best_info['accuracy'] + 0.002 else "Tidak",
                   f"acc={ord_results['oof_accuracy']:.4f}")

    if ord_results['oof_accuracy'] > best_info['accuracy'] + 0.002:
        best_info.update({'accuracy': ord_results['oof_accuracy'],
                           'macro_f1': ord_results['macro_f1'],
                           'model': 'OrdinalCumulative', 'exp_id': eid})
        save_checkpoint(eid, 'OrdinalCumulative', ord_results, 'full_advanced', 'HGB 3x')
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if ord_test_probs is not None:
        ord_test_preds = np.argmax(ord_test_probs, axis=1)
        make_submission(ord_test_preds, sample_sub, f"submission_{eid}.csv")

    # Ordinal with regression approach
    print("\n--- Ordinal Regression Model ---")
    from sklearn.linear_model import Ridge
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_adv)
    X_t_scaled = scaler.transform(X_t_adv)

    # Train Ridge regression to predict continuous target
    experiment_count += 1
    eid = f"EXP-{experiment_count:03d}"

    reg_oof = np.zeros(len(y))
    reg_test_list = []
    reg_fold_accs = []

    for tr, val in cv.split(X_scaled, y):
        X_tr, X_val = X_scaled[tr], X_scaled[val]
        y_tr, y_val = y[tr], y[val]

        reg = Ridge(alpha=1.0, random_state=SEED)
        reg.fit(X_tr, y_tr)
        reg_pred = reg.predict(X_val)
        reg_test_list.append(reg.predict(X_t_scaled))

        # Convert to class: round to nearest class
        reg_class = np.round(reg_pred).clip(0, 3).astype(int)
        reg_oof[val] = reg_class
        reg_fold_accs.append(accuracy_score(y_val, reg_class))

    reg_acc = accuracy_score(y, reg_oof)
    reg_f1 = f1_score(y, reg_oof, average='macro')
    reg_metrics = {
        'mean_accuracy': float(np.mean(reg_fold_accs)),
        'std_accuracy': float(np.std(reg_fold_accs)),
        'min_fold': float(np.min(reg_fold_accs)),
        'oof_accuracy': float(reg_acc),
        'macro_f1': float(reg_f1),
        'oof_predictions': reg_oof,
        'oof_probabilities': np.zeros((len(y), 4)),
        'overfit_gap': 0, 'train_score': 0, 'runtime': 0,
        'balanced_accuracy': float(balanced_accuracy_score(y, reg_oof)),
    }

    log_experiment(eid, best_info['exp_id'],
                   "Ordinal regression (Ridge) + rounding threshold",
                   "full_advanced", "OrdinalRidge", "{alpha:1.0}",
                   "RepeatedStratifiedKFold(5,2)", SEED, reg_metrics,
                   "Ya" if reg_acc > best_info['accuracy'] + 0.002 else "Tidak",
                   f"acc={reg_acc:.4f}")
    print(f"  OrdinalRidge — OOF: {reg_acc:.4f}, F1={reg_f1:.4f}")

    if reg_acc > best_info['accuracy'] + 0.002:
        best_info.update({'accuracy': reg_acc, 'macro_f1': reg_f1,
                           'model': 'OrdinalRidge', 'exp_id': eid})
        save_checkpoint(eid, 'OrdinalRidge', reg_metrics, 'full_advanced', '{alpha:1.0}')
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    reg_test_preds = np.round(np.mean(reg_test_list, axis=0)).clip(0, 3).astype(int)
    make_submission(reg_test_preds, sample_sub, f"submission_{eid}.csv")

    # ============================================================
    # SUB-PHASE: FEATURE SELECTION
    # ============================================================
    print(f"\n{'='*60}")
    print("SUB-PHASE: FEATURE SELECTION")
    print(f"{'='*60}")

    X_base, X_test_base = fsets['full_advanced']
    n_features = X_base.shape[1]

    for n_sel in [20, 30, 50]:
        try:
            selector = SelectKBest(mutual_info_classif, k=min(n_sel, n_features))
            selector.fit(X_base, y)
            mask = selector.get_support()
            X_sel = X_base[:, mask]
            X_test_sel = X_test_base[:, mask]
            print(f"\n  Top {n_sel} features (MI): {X_sel.shape}")

            experiment_count += 1
            eid = f"EXP-{experiment_count:03d}"
            run_exp(lambda rs: SVC(C=10, gamma='scale', probability=True, random_state=rs),
                    f"SVC_MI_{n_sel}", f"MI_top_{n_sel}", X_sel, y, X_test_sel, sample_sub,
                    best_info['exp_id'], f"SVC with top {n_sel} MI features")
        except Exception as e:
            print(f"  MI selection failed for k={n_sel}: {e}")

    # ANOVA selection
    for n_sel in [20, 30]:
        try:
            selector = SelectKBest(f_classif, k=n_sel)
            selector.fit(X_base, y)
            mask = selector.get_support()
            X_sel = X_base[:, mask]
            X_test_sel = X_test_base[:, mask]

            run_exp(lambda rs: SVC(C=10, gamma='scale', probability=True, random_state=rs),
                    f"SVC_ANOVA_{n_sel}", f"ANOVA_top_{n_sel}",
                    X_sel, y, X_test_sel, sample_sub,
                    best_info['exp_id'], f"SVC with top {n_sel} ANOVA features")
        except Exception as e:
            print(f"  ANOVA failed: {e}")

    # ============================================================
    # SUB-PHASE: HYPERPARAMETER TUNING
    # ============================================================
    print(f"\n{'='*60}")
    print("SUB-PHASE: HYPERPARAMETER TUNING (Optuna)")
    print(f"{'='*60}")

    X_adv, X_t_adv = fsets['full_advanced']

    # SVC tuning
    print("\n--- Tuning SVC RBF ---")
    best_svc_params, best_svc_score = optuna_tune(X_adv, y, 'svc', n_trials=30)
    experiment_count += 1
    eid = f"EXP-{experiment_count:03d}"
    tuned_svc = SVC(**best_svc_params, probability=True, random_state=SEED)
    res = evaluate_model_cv(X_adv, y, tuned_svc, "SVC_tuned", X_test=X_t_adv, seed=SEED, n_repeats=2)
    log_experiment(eid, best_info['exp_id'], f"SVC tuned with Optuna: {best_svc_params}",
                   "full_advanced", "SVC_tuned", str(best_svc_params),
                   "RepeatedStratifiedKFold(5,2)", SEED, res,
                   "Ya" if res['oof_accuracy'] > best_info['accuracy'] + 0.002 else "Tidak", "")
    if res['oof_accuracy'] > best_info['accuracy'] + 0.002:
        best_info.update({'accuracy': res['oof_accuracy'], 'macro_f1': res['macro_f1'],
                          'model': 'SVC_tuned', 'exp_id': eid})
        save_checkpoint(eid, 'SVC_tuned', res, 'full_advanced', str(best_svc_params))
        no_improvement_count = 0
    else:
        no_improvement_count += 1
    if 'test_preds' in res:
        make_submission(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, res['test_preds']),
                        sample_sub, f"submission_{eid}.csv")

    # XGBoost tuning
    try:
        print("\n--- Tuning XGBoost ---")
        best_xgb_params, best_xgb_score = optuna_tune(X_adv, y, 'xgb', n_trials=30)
        import xgboost as xgb
        experiment_count += 1
        eid = f"EXP-{experiment_count:03d}"
        tuned_xgb = xgb.XGBClassifier(**best_xgb_params, random_state=SEED, n_jobs=-1, verbosity=0)
        res = evaluate_model_cv(X_adv, y, tuned_xgb, "XGB_tuned", X_test=X_t_adv, seed=SEED, n_repeats=2)
        log_experiment(eid, best_info['exp_id'], f"XGBoost tuned",
                       "full_advanced", "XGB_tuned", str(best_xgb_params),
                       "RepeatedStratifiedKFold(5,2)", SEED, res,
                       "Ya" if res['oof_accuracy'] > best_info['accuracy'] + 0.002 else "Tidak", "")
        if res['oof_accuracy'] > best_info['accuracy'] + 0.002:
            best_info.update({'accuracy': res['oof_accuracy'], 'macro_f1': res['macro_f1'],
                              'model': 'XGB_tuned', 'exp_id': eid})
            save_checkpoint(eid, 'XGB_tuned', res, 'full_advanced', str(best_xgb_params))
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if 'test_preds' in res:
            make_submission(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, res['test_preds']),
                            sample_sub, f"submission_{eid}.csv")
    except Exception as e:
        print(f"  XGBoost tuning failed: {e}")

    # LightGBM tuning
    try:
        print("\n--- Tuning LightGBM ---")
        best_lgb_params, best_lgb_score = optuna_tune(X_adv, y, 'lgb', n_trials=30)
        import lightgbm as lgb
        experiment_count += 1
        eid = f"EXP-{experiment_count:03d}"
        tuned_lgb = lgb.LGBMClassifier(**best_lgb_params, random_state=SEED, n_jobs=-1, verbose=-1)
        res = evaluate_model_cv(X_adv, y, tuned_lgb, "LGB_tuned", X_test=X_t_adv, seed=SEED, n_repeats=2)
        log_experiment(eid, best_info['exp_id'], f"LightGBM tuned",
                       "full_advanced", "LGB_tuned", str(best_lgb_params),
                       "RepeatedStratifiedKFold(5,2)", SEED, res,
                       "Ya" if res['oof_accuracy'] > best_info['accuracy'] + 0.002 else "Tidak", "")
        if res['oof_accuracy'] > best_info['accuracy'] + 0.002:
            best_info.update({'accuracy': res['oof_accuracy'], 'macro_f1': res['macro_f1'],
                              'model': 'LGB_tuned', 'exp_id': eid})
            save_checkpoint(eid, 'LGB_tuned', res, 'full_advanced', str(best_lgb_params))
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if 'test_preds' in res:
            make_submission(np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, res['test_preds']),
                            sample_sub, f"submission_{eid}.csv")
    except Exception as e:
        print(f"  LightGBM tuning failed: {e}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {experiment_count}")
    print(f"Best: {best_info['model']} = {best_info['accuracy']:.4f} (F1={best_info['macro_f1']:.4f})")
    print(f"No-improvement count: {no_improvement_count}")

    summary = {
        'phase': 2,
        'experiments_run': experiment_count,
        'best_accuracy': best_info['accuracy'],
        'best_macro_f1': best_info['macro_f1'],
        'best_model': best_info['model'],
        'best_feature_set': best_info.get('feature_set', ''),
        'no_improvement_count': no_improvement_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(EXP_DIR / "reports" / "phase2_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    if best_info['accuracy'] >= 0.70:
        print("\n*** TARGET 0.70 ACHIEVED! ***")
    else:
        improvement = best_info['accuracy'] - 0.5156
        print(f"\nImprovement from baseline: +{improvement:.4f}")
        print(f"Remaining to 0.70: {0.70 - best_info['accuracy']:.4f}")
        print("Continue to Phase 3: Ensemble + Residual Modeling + Confidence Switching")

if __name__ == "__main__":
    main()
