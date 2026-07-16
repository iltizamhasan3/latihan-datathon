"""
Core forensic utilities: lockbox, CV, multi-seed, nested CV, logging.
All experiments use leakage-safe preprocessing.
"""
import numpy as np, pandas as pd, json, os, time, sys, warnings, copy, pickle
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                      train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from experiments.features import get_all_features, engineer_features, WEEK_COLS, ACTIVITY_COLS, BEHAVIORAL_COLS, EXAM_COLS, TASK_COLS, DEMO_COLS

SEED = 42
DATA_DIR = Path("data")
EXP_DIR = Path("experiments")
FOR_DIR = Path("forensic_experiments")

RANDOM_SEEDS = [42, 123, 2026, 3407, 7777]

def load_data():
    t = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    s = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return t, te, s

def get_cv(rs=SEED):
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=rs)

def get_skf(rs=SEED):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)

# ====== LOCKBOX ======
def create_lockbox(train, y, test_size=0.20, random_state=20260715):
    """Create a locked holdout set - never used for any decision."""
    lock_train_idx, lock_val_idx = train_test_split(
        np.arange(len(train)), test_size=test_size,
        stratify=y, random_state=random_state
    )
    lockbox_info = {
        'train_indices': lock_train_idx.tolist(),
        'val_indices': lock_val_idx.tolist(),
        'test_size': test_size,
        'random_state': random_state,
        'train_n': len(lock_train_idx),
        'val_n': len(lock_val_idx),
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'opened': 0,
        'opened_for': []
    }
    # Save lockbox info
    lf = FOR_DIR / "lockbox" / "lockbox_info.json"
    with open(lf, 'w') as f: json.dump(lockbox_info, f, indent=2)
    print(f"Lockbox created: {len(lock_train_idx)} train, {len(lock_val_idx)} holdout")
    return lock_train_idx, lock_val_idx, lockbox_info

def open_lockbox(candidate_name):
    """Mark that lockbox was opened for a candidate. Max 5 openings."""
    lf = FOR_DIR / "lockbox" / "lockbox_info.json"
    with open(lf) as f: info = json.load(f)
    info['opened'] += 1
    info['opened_for'].append({'candidate': candidate_name, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    if info['opened'] > 5:
        print("WARNING: Lockbox opened more than 5 times!")
    with open(lf, 'w') as f: json.dump(info, f, indent=2)
    return info['train_indices'], info['val_indices']

def evaluate_lockbox(model_fn, X, y, train_idx, val_idx):
    """Evaluate on locked holdout. model_fn(scaler, X_train, y_train) -> fitted model."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    m = model_fn(42)
    m.fit(X_tr, y[train_idx])
    preds = m.predict(X_val)
    acc = accuracy_score(y[val_idx], preds)
    f1 = f1_score(y[val_idx], preds, average='macro')
    print(f"  Lockbox: {acc:.4f} F1={f1:.4f}")
    return acc, f1, preds

# ====== CORE EVAL ======
def evaluate_cv(X, y, model_fn, name, X_test=None, seed=SEED, use_skf=False):
    """Evaluate model with CV. model_fn(seed) -> fresh model instance.
    Returns rich metrics dict.
    """
    cv = get_skf(seed) if use_skf else get_cv(seed)
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
        m = model_fn(seed + fi)
        m.fit(X_tr, y[tr])

        oof_preds[val] = np.ravel(m.predict(X_val))
        if hasattr(m, 'predict_proba'):
            oof_probs[val] = m.predict_proba(X_val)

        tr_preds = m.predict(X_tr)
        tr_acc = accuracy_score(y[tr], tr_preds)
        val_acc = accuracy_score(y[val], oof_preds[val])
        val_f1 = f1_score(y[val], oof_preds[val], average='macro')
        val_bal = balanced_accuracy_score(y[val], oof_preds[val])

        fold_metrics.append({
            'fold': fi, 'val_accuracy': val_acc, 'val_f1': val_f1,
            'val_balanced': val_bal, 'train_accuracy': tr_acc,
            'gap': tr_acc - val_acc
        })

        if X_test is not None:
            te_preds = m.predict(scaler.transform(X_test))
            test_preds_list.append(te_preds)

    elapsed = time.time() - start

    # Aggregate
    val_accs = [m['val_accuracy'] for m in fold_metrics]
    oof_acc = accuracy_score(y, oof_preds)
    oof_f1 = f1_score(y, oof_preds, average='macro')
    oof_bal = balanced_accuracy_score(y, oof_preds)

    # Confusion matrix
    cm = confusion_matrix(y, oof_preds)

    metrics = {
        'oof_accuracy': float(oof_acc),
        'mean_fold_accuracy': float(np.mean(val_accs)),
        'std_accuracy': float(np.std(val_accs)),
        'min_fold': float(np.min(val_accs)),
        'max_fold': float(np.max(val_accs)),
        'macro_f1': float(oof_f1),
        'balanced_accuracy': float(oof_bal),
        'mean_train_accuracy': float(np.mean([m['train_accuracy'] for m in fold_metrics])),
        'mean_gap': float(np.mean([m['gap'] for m in fold_metrics])),
        'runtime': elapsed,
        'oof_predictions': oof_preds,
        'oof_probabilities': oof_probs,
        'fold_metrics': fold_metrics,
        'confusion_matrix': cm.tolist(),
    }

    if test_preds_list:
        metrics['test_preds'] = np.array(test_preds_list)

    print(f"  {name:40s} OOF={oof_acc:.4f} F1={oof_f1:.4f} min={np.min(val_accs):.4f} "
          f"gap={metrics['mean_gap']:.3f} {elapsed:.0f}s")
    return metrics

def evaluate_cv_simple(X, y, model_fn, name, seed=SEED):
    """Quick CV evaluation without full metrics dict - for fast iteration."""
    cv = get_cv(seed)
    n = len(y)
    oof_preds = np.zeros(n, dtype=int)
    fold_accs = []
    for fi, (tr, val) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_val = scaler.transform(X[val])
        m = model_fn(seed + fi)
        m.fit(X_tr, y[tr])
        oof_preds[val] = m.predict(X_val)
        fold_accs.append(accuracy_score(y[val], oof_preds[val]))
    oof_acc = accuracy_score(y, oof_preds)
    return oof_acc, oof_preds, np.mean(fold_accs), np.std(fold_accs)

# ====== MULTI-SEED VALIDATION ======
def multi_seed_validate(X, y, model_fn, name, seeds=RANDOM_SEEDS):
    """Validate model across multiple seeds."""
    results = {}
    all_oof = {}
    for seed in seeds:
        results[seed] = evaluate_cv(X, y, lambda s: model_fn(s), f"{name}_s{seed}", seed=seed)
        all_oof[seed] = results[seed]['oof_predictions']

    accs = [results[s]['oof_accuracy'] for s in seeds]
    f1s = [results[s]['macro_f1'] for s in seeds]
    min_folds = [results[s]['min_fold'] for s in seeds]

    summary = {
        'model': name,
        'seeds_tested': seeds,
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'median_accuracy': float(np.median(accs)),
        'min_seed_accuracy': float(np.min(accs)),
        'max_seed_accuracy': float(np.max(accs)),
        'mean_f1': float(np.mean(f1s)),
        'mean_min_fold': float(np.mean(min_folds)),
        'per_seed': {str(s): {'accuracy': results[s]['oof_accuracy'],
                              'f1': results[s]['macro_f1'],
                              'min_fold': results[s]['min_fold']} for s in seeds}
    }

    # Agreement across seeds
    agreement_matrix = np.zeros((len(seeds), len(seeds)))
    for i, s1 in enumerate(seeds):
        for j, s2 in enumerate(seeds):
            agreement_matrix[i, j] = np.mean(all_oof[s1] == all_oof[s2])
    summary['mean_seed_agreement'] = float(np.mean([agreement_matrix[i,j]
        for i in range(len(seeds)) for j in range(i+1, len(seeds))]))

    print(f"\n  Multi-seed ({name}): mean={summary['mean_accuracy']:.4f} "
          f"std={summary['std_accuracy']:.4f} agreement={summary['mean_seed_agreement']:.4f}")
    for s in seeds:
        print(f"    Seed {s}: {results[s]['oof_accuracy']:.4f}")

    return summary, results, all_oof

# ====== NESTED CV ======
def nested_cv_eval(X, y, outer_model_fn, inner_model_fn_or_tune, name, n_outer_splits=5, seed=2026):
    """Nested CV: outer loop scores, inner loop for tuning/selection.
    inner_model_fn_or_tune: callable (X_train, y_train, X_val, y_val) -> tuned model
    """
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=seed)
    n = len(y)
    outer_preds = np.zeros(n, dtype=int)
    outer_probs = np.zeros((n, 4))
    outer_scores = []
    all_configs = []

    for oi, (otr, oval) in enumerate(outer_cv.split(X, y)):
        print(f"\n  Outer fold {oi+1}/{n_outer_splits}: train={len(otr)} val={len(oval)}")

        # Inner CV on otr
        scaler = StandardScaler()
        X_otr = scaler.fit_transform(X[otr])
        X_oval = scaler.transform(X[oval])

        if inner_model_fn_or_tune is not None:
            # Inner loop: tune/select model
            best_model = inner_model_fn_or_tune(X[otr], y[otr], X[oval], y[oval])
        else:
            best_model = outer_model_fn(seed + oi)

        best_model.fit(X_otr, y[otr])
        outer_preds[oval] = best_model.predict(X_oval)
        if hasattr(best_model, 'predict_proba'):
            outer_probs[oval] = best_model.predict_proba(X_oval)
        outer_scores.append(accuracy_score(y[oval], outer_preds[oval]))

    nested_acc = accuracy_score(y, outer_preds)
    nested_f1 = f1_score(y, outer_preds, average='macro')
    print(f"\n  Nested CV: {name}: acc={nested_acc:.4f} F1={nested_f1:.4f} "
          f"mean_outer={np.mean(outer_scores):.4f}")

    return {
        'nested_accuracy': float(nested_acc),
        'nested_f1': float(nested_f1),
        'outer_fold_scores': [float(s) for s in outer_scores],
        'mean_outer_fold': float(np.mean(outer_scores)),
        'std_outer_fold': float(np.std(outer_scores))
    }

# ====== LOGGING ======
def log_experiment(eid, parent, hyp, fset, model, params, metrics, accepted, notes=""):
    """Log experiment to forensic experiment_log.csv"""
    p = FOR_DIR / "experiments" / "experiment_log.csv"
    oof_acc = metrics.get('oof_accuracy', metrics.get('mean_accuracy', 0))
    row = {
        'experiment_id': eid, 'parent_experiment': parent or '',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hypothesis': hyp[:150], 'feature_set': fset, 'model': model,
        'parameters': str(params)[:200],
        'cv_strategy': 'RSKF(5,2)', 'seed': SEED,
        'mean_accuracy': f"{oof_acc:.6f}",
        'std_accuracy': f"{metrics.get('std_accuracy', 0):.6f}",
        'minimum_fold': f"{metrics.get('min_fold', 0):.6f}",
        'macro_f1': f"{metrics.get('macro_f1', 0):.6f}",
        'balanced_accuracy': f"{metrics.get('balanced_accuracy', 0):.6f}",
        'train_score': f"{metrics.get('mean_train_accuracy', 0):.6f}",
        'overfit_gap': f"{metrics.get('mean_gap', 0):.6f}",
        'runtime': f"{metrics.get('runtime', 0):.2f}",
        'accepted': accepted, 'notes': notes[:200]
    }
    pd.DataFrame([row]).to_csv(p, mode='a', header=not p.exists(), index=False)

def update_leaderboard():
    """Update leaderboard from experiment log."""
    p = FOR_DIR / "experiments" / "experiment_log.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    if df.empty: return
    df['acc'] = pd.to_numeric(df['mean_accuracy'], errors='coerce')
    top = df.sort_values('acc', ascending=False).head(20)
    top.to_csv(FOR_DIR / "experiments" / "leaderboard.csv", index=False)
    return top

def checkpoint(eid, model, metrics, fset, params):
    """Save best model checkpoint."""
    ckpt = {
        'experiment_id': eid, 'model': model, 'feature_set': fset,
        'parameters': params,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'mean_accuracy': metrics.get('oof_accuracy', 0),
            'std_accuracy': metrics.get('std_accuracy', 0),
            'min_fold': metrics.get('min_fold', 0),
            'macro_f1': metrics.get('macro_f1', 0),
            'balanced_accuracy': metrics.get('balanced_accuracy', 0),
            'overfit_gap': metrics.get('mean_gap', 0),
        }
    }
    with open(FOR_DIR / "checkpoints" / "best_config.json", 'w') as f:
        json.dump(ckpt, f, indent=2)
    np.save(FOR_DIR / "oof_predictions" / "best_oof_preds.npy", metrics['oof_predictions'])
    np.save(FOR_DIR / "oof_predictions" / "best_oof_probs.npy", metrics['oof_probabilities'])
    pd.DataFrame({
        'id': range(len(metrics['oof_predictions'])),
        'target': metrics['oof_predictions']
    }).to_csv(FOR_DIR / "oof_predictions" / "best_oof.csv", index=False)
    print(f"  >>> CHECKPOINT: {eid} {model} = {metrics.get('oof_accuracy',0):.4f}")

def make_submission(preds, template, name):
    """Generate submission file."""
    if isinstance(preds, np.ndarray) and preds.ndim > 1 and preds.shape[0] > 1 and preds.shape[1] > 1:
        # Ensemble of multiple predictions: majority vote
        from scipy import stats
        preds = stats.mode(preds, axis=0)[0].ravel()
    s = template.copy()
    s['target'] = np.ravel(preds).astype(int)
    s.to_csv(FOR_DIR / "submissions" / name, index=False)

# ====== BUILD FEATURES ======
def build_base_features(train, test):
    """Build standard feature set: engineered + PCA + cluster."""
    X = get_all_features(train)
    Xt = get_all_features(test)

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # PCA on weekly
    for gname, cols, nc in [('wp', WEEK_COLS, 5), ('ap', ACTIVITY_COLS, 6)]:
        s = StandardScaler()
        d = s.fit_transform(train[cols].fillna(0))
        dt = s.transform(test[cols].fillna(0))
        p = PCA(n_components=nc, random_state=SEED)
        for i, c in enumerate(p.fit_transform(d).T):
            X[f'{gname}_{i+1}'] = c
        for i, c in enumerate(p.transform(dt).T):
            Xt[f'{gname}_{i+1}'] = c

    # KMeans distance features
    for cname, cols, nc in [('wc', WEEK_COLS, 5), ('ac', ACTIVITY_COLS, 5)]:
        s = StandardScaler()
        d = s.fit_transform(train[cols].fillna(0))
        dt = s.transform(test[cols].fillna(0))
        k = KMeans(n_clusters=nc, random_state=SEED, n_init=10)
        k.fit(d)
        for i, c in enumerate(k.transform(d).T):
            X[f'{cname}_cd{i+1}'] = c
        for i, c in enumerate(k.transform(dt).T):
            Xt[f'{cname}_cd{i+1}'] = c

    return X.fillna(0).replace([np.inf, -np.inf], 0), \
           Xt.fillna(0).replace([np.inf, -np.inf], 0)

# ====== REPRODUCIBILITY ======
def compute_score_065_status():
    """Determine 0.65 score status based on reproducibility experiments."""
    report_path = FOR_DIR / "audit" / "score_065_audit.md"
    status_path = FOR_DIR / "audit" / "score_065_status.json"
    if status_path.exists():
        with open(status_path) as f:
            return json.load(f)['status']
    return "NOT_YET_DETERMINED"

def save_status(status, details):
    """Save score_065 status."""
    status_path = FOR_DIR / "audit" / "score_065_status.json"
    with open(status_path, 'w') as f:
        json.dump({'status': status, 'details': details,
                   'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

if __name__ == "__main__":
    print("Forensic core utilities loaded.")
    print(f"Seeds: {RANDOM_SEEDS}")
    train, test, sample = load_data()
    y = train['target'].values
    create_lockbox(train, y)
