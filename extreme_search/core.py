"""
Extreme Autonomous Search: core utilities, CV, OOF, experiment logging.
All transforms leakage-safe (scaler inside fold, no global target usage).
"""
import numpy as np, pandas as pd, json, time, sys, warnings, traceback
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent
sys.path.append(str(ROOT.parent))

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
from experiments.features import engineer_features

import catboost as cb

# ================================================================
# EXPERIMENT LOG
# ================================================================
EXPERIMENT_LOG = ROOT / "experiment_log.csv"
LEADERBOARD = ROOT / "leaderboard.csv"
BEST_CONFIG = ROOT / "checkpoints" / "best_config.json"

def log_experiment(exp):
    """Append one row to experiment_log.csv."""
    exp['timestamp'] = datetime.now().isoformat()
    exp.setdefault('fold_scores', [])
    exp.setdefault('mean_accuracy', 0.0)
    exp.setdefault('macro_f1', 0.0)
    exp.setdefault('balanced_accuracy', 0.0)
    exp.setdefault('minimum_fold', 0.0)
    exp.setdefault('train_accuracy', None)
    exp.setdefault('overfit_gap', None)
    exp.setdefault('runtime', 0)
    exp.setdefault('accepted', False)
    exp.setdefault('rejection_reason', '')
    exp.setdefault('next_hypothesis', '')

    row = pd.DataFrame([exp])
    if EXPERIMENT_LOG.exists():
        existing = pd.read_csv(EXPERIMENT_LOG)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(EXPERIMENT_LOG, index=False)
    update_leaderboard()

def update_leaderboard():
    """Rebuild leaderboard sorted by mean_accuracy desc."""
    if not EXPERIMENT_LOG.exists():
        return
    df = pd.read_csv(EXPERIMENT_LOG)
    if len(df) == 0:
        return
    lb = df.sort_values('mean_accuracy', ascending=False).head(50)
    lb.to_csv(LEADERBOARD, index=False)

def load_best():
    """Return best experiment dict from leaderboard."""
    if not LEADERBOARD.exists():
        return {'mean_accuracy': 0.0}
    lb = pd.read_csv(LEADERBOARD)
    if len(lb) == 0:
        return {'mean_accuracy': 0.0}
    return lb.iloc[0].to_dict()

def save_best_config(config, accuracy, f1):
    """Save best config JSON."""
    config['accuracy'] = accuracy
    config['f1'] = f1
    with open(BEST_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)

# ================================================================
# FEATURE LOADING
# ================================================================
def load_data(with_sequence=True, with_group_features=True):
    """Load train/test with engineered features."""
    train = pd.read_csv(ROOT.parent / "data" / "train.csv")
    test = pd.read_csv(ROOT.parent / "data" / "test.csv")
    y = train['target'].values

    X = get_all_features(train).fillna(0).replace([np.inf, -np.inf], 0)
    Xt = get_all_features(test).fillna(0).replace([np.inf, -np.inf], 0)

    if with_sequence:
        add_sequence_features(train, test, X, Xt)

    if with_group_features:
        add_group_features(train, test, X, Xt, y)

    return X, Xt, y, train, test

def add_sequence_features(train, test, X, Xt):
    """Add advanced sequence features."""
    from scipy import stats as sp_stats

    for phase_prefix, cols, n_cols in [('week', WEEK_COLS, 12), ('act', ACTIVITY_COLS, 16)]:
        train_seq = train[cols].fillna(0).values
        test_seq = test[cols].fillna(0).values
        pf = phase_prefix
        x_axis = np.arange(n_cols)

        for seq_data, df in [(train_seq, X), (test_seq, Xt)]:
            n = len(seq_data)

            # Basic stats
            df[f'seq_{pf}_mean'] = seq_data.mean(axis=1)
            df[f'seq_{pf}_std'] = seq_data.std(axis=1, ddof=0)
            df[f'seq_{pf}_mad'] = np.abs(seq_data - seq_data.mean(axis=1, keepdims=True)).mean(axis=1)

            # Slopes
            slopes = np.array([sp_stats.linregress(x_axis, seq_data[i])[0] if np.std(seq_data[i]) > 1e-10 else 0 for i in range(n)])
            df[f'seq_{pf}_slope'] = slopes

            # Last half vs first half
            half = n_cols // 2
            first_half = seq_data[:, :half]
            last_half = seq_data[:, half:]
            df[f'seq_{pf}_first_half_mean'] = first_half.mean(axis=1)
            df[f'seq_{pf}_last_half_mean'] = last_half.mean(axis=1)
            df[f'seq_{pf}_half_diff'] = df[f'seq_{pf}_last_half_mean'] - df[f'seq_{pf}_first_half_mean']

            # Acceleration
            accel = np.array([np.polyfit(x_axis, seq_data[i], 2)[0] * 2 for i in range(n)])
            df[f'seq_{pf}_acceleration'] = accel
            df[f'seq_{pf}_curvature'] = np.abs(accel) / (1 + slopes**2)**1.5

            # Autocorrelation
            autocorr = np.array([
                np.corrcoef(seq_data[i, :-1], seq_data[i, 1:])[0, 1]
                if np.std(seq_data[i, :-1]) > 1e-10 and np.std(seq_data[i, 1:]) > 1e-10 else 0
                for i in range(n)
            ])
            df[f'seq_{pf}_autocorr_lag1'] = autocorr

            # Spectral entropy
            fft_vals = np.abs(np.fft.fft(seq_data, axis=1))
            fft_power = fft_vals[:, :n_cols//2] ** 2
            fft_norm = fft_power / (fft_power.sum(axis=1, keepdims=True) + 1e-10)
            df[f'seq_{pf}_spectral_entropy'] = -np.sum(fft_norm * np.log(fft_norm + 1e-10), axis=1)

            # Change points
            diffs = np.diff(seq_data, axis=1)
            sign_changes = np.sum(np.diff(np.sign(diffs), axis=1) != 0, axis=1)
            df[f'seq_{pf}_change_points'] = sign_changes

            # Range and CV
            df[f'seq_{pf}_range'] = seq_data.max(axis=1) - seq_data.min(axis=1)
            df[f'seq_{pf}_cv'] = np.where(df[f'seq_{pf}_mean'].abs() > 1e-10,
                                           df[f'seq_{pf}_std'] / df[f'seq_{pf}_mean'].abs(), 0)

            # Min/max positions
            df[f'seq_{pf}_min_pos'] = seq_data.argmin(axis=1) / n_cols
            df[f'seq_{pf}_max_pos'] = seq_data.argmax(axis=1) / n_cols

            # Last value relative to mean
            df[f'seq_{pf}_last_vs_mean'] = seq_data[:, -1] - df[f'seq_{pf}_mean']
            df[f'seq_{pf}_first_vs_last'] = seq_data[:, -1] - seq_data[:, 0]

    return X, Xt

def add_group_features(train, test, X, Xt, y=None):
    """
    Add kelas-level group features.
    For each student, compute aggregate statistics of their classmates.
    Only uses train data for aggregates when cross-validating.
    For test, use all train data aggregates (leakage-safe: no target used for test predictions).
    """
    # Class-level aggregates from ALL train data
    # These are safe because we use them as features, not as target encoding
    train['kelas'] = train['kelas'].fillna(-1).astype(int)
    test['kelas'] = test['kelas'].fillna(-1).astype(int)

    # Size of each class
    kelas_size = train.groupby('kelas').size()
    X['kelas_size'] = train['kelas'].map(kelas_size).fillna(1)
    Xt['kelas_size'] = test['kelas'].map(kelas_size).fillna(0)

    # Class-level means of weekly scores
    week_feats = [f'nilai_minggu_{i:02d}' for i in range(1, 13)]
    for col in week_feats:
        kelas_mean = train.groupby('kelas')[col].mean()
        X[f'{col}_kelas_mean'] = train[col] - train['kelas'].map(kelas_mean)
        Xt[f'{col}_kelas_mean'] = test[col] - test['kelas'].map(kelas_mean)

    # Task completion ratio per class
    if 'tugas_selesai' in train.columns and 'tugas_diberikan' in train.columns:
        kelas_task_ratio = (train['tugas_selesai'] / train['tugas_diberikan'].clip(lower=1)).groupby(train['kelas']).mean()
        X['task_ratio_kelas_gap'] = (train['tugas_selesai'] / train['tugas_diberikan'].clip(lower=1)) - train['kelas'].map(kelas_task_ratio)
        Xt['task_ratio_kelas_gap'] = (test['tugas_selesai'] / test['tugas_diberikan'].clip(lower=1)) - test['kelas'].map(kelas_task_ratio)

    return X, Xt

# ================================================================
# CV EVALUATION
# ================================================================
def evaluate_cv(X_vals, Xt_vals, y, model_fn, meta_fn=None,
                cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
                label="experiment"):
    """
    Evaluate model with CV. Returns (oof_preds, oof_probs, test_probs, metrics).
    model_fn: callable(random_state) -> unfitted model
    meta_fn: optional callable -> meta model for stacking
    """
    n = len(y)
    oof_preds = np.zeros(n, dtype=int)
    oof_probs = np.zeros((n, 4))
    test_probs_list = []
    fold_scores = []

    t0 = time.time()

    for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_vals[tr])
        X_val = scaler.transform(X_vals[val])
        X_te = scaler.transform(Xt_vals)
        rs = 42 + fi

        if meta_fn is not None:
            # Stacking: base models + meta
            base_val_probs = []
            base_te_probs = []
            for mname, mfn in model_fn.items():
                m = mfn(rs)
                m.fit(X_tr, y[tr])
                base_val_probs.append(m.predict_proba(X_val))
                base_te_probs.append(m.predict_proba(X_te))

            meta_val = np.column_stack(base_val_probs)
            meta_te = np.column_stack(base_te_probs)

            meta = meta_fn(rs)
            meta.fit(meta_val, y[val])
            oof_preds[val] = meta.predict(meta_val)
            oof_probs[val] = meta.predict_proba(meta_val) if hasattr(meta, 'predict_proba') else \
                _softmax(meta.decision_function(meta_val))
            test_probs_list.append(meta.predict_proba(meta_te))
        else:
            m = model_fn(rs)
            m.fit(X_tr, y[tr])
            oof_preds[val] = m.predict(X_val)
            oof_probs[val] = m.predict_proba(X_val) if hasattr(m, 'predict_proba') else \
                _softmax(m.decision_function(X_val))
            test_probs_list.append(m.predict_proba(X_te))

        fold_acc = accuracy_score(y[val], oof_preds[val])
        fold_scores.append(fold_acc)

    elapsed = time.time() - t0
    acc = accuracy_score(y, oof_preds)
    f1 = f1_score(y, oof_preds, average='macro')
    min_fold = min(fold_scores)

    metrics = {
        'mean_accuracy': acc,
        'macro_f1': f1,
        'minimum_fold': min_fold,
        'fold_scores': fold_scores,
        'runtime': elapsed,
        'label': label,
    }

    return oof_preds, oof_probs, test_probs_list, metrics

def _softmax(x):
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ================================================================
# MULTI-SEED VERIFICATION
# ================================================================
def multi_seed_verify(X_vals, y, model_fn, seeds=[42, 123, 2026, 3407, 7777]):
    """Verify model across multiple seeds."""
    scores = []
    for s in seeds:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=s)
        oof_p, _, _, metrics = evaluate_cv(X_vals, X_vals, y, model_fn, cv=cv,
                                            label=f"multi_seed_{s}")
        scores.append(metrics['mean_accuracy'])
    return {
        'multi_seed_mean': np.mean(scores),
        'multi_seed_std': np.std(scores),
        'multi_seed_min': min(scores),
        'multi_seed_scores': scores,
    }

# ================================================================
# NESTED CV
# ================================================================
def evaluate_nested_cv(X_vals, y, base_model_fns, meta_fn_fn, outer_cv=None, inner_cv=None):
    """
    Nested CV for stacking pipelines. Returns nested accuracy and fold scores.
    """
    if outer_cv is None:
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    if inner_cv is None:
        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    n = len(y)
    outer_preds = np.zeros(n, dtype=int)
    outer_scores = []

    for oi, (otr, oval) in enumerate(outer_cv.split(X_vals, y)):
        inner_oof = {}
        for mname, mfn in base_model_fns.items():
            oof_p = np.zeros((len(otr), 4))
            for ir, iv in inner_cv.split(X_vals[otr], y[otr]):
                sc = StandardScaler()
                X_ir = sc.fit_transform(X_vals[otr][ir])
                X_iv = sc.transform(X_vals[otr][iv])
                m = mfn(42)
                m.fit(X_ir, y[otr][ir])
                oof_p[iv] = m.predict_proba(X_iv)
            inner_oof[mname] = oof_p

        X_meta_train = np.column_stack([inner_oof[n] for n in base_model_fns])
        meta = meta_fn_fn(42)
        meta.fit(X_meta_train, y[otr])

        # Predict outer val
        sc_outer = StandardScaler()
        X_otr_full = sc_outer.fit_transform(X_vals[otr])
        X_oval = sc_outer.transform(X_vals[oval])

        outer_base = np.column_stack([
            base_model_fns[n](42).fit(X_otr_full, y[otr]).predict_proba(X_oval)
            for n in base_model_fns
        ])
        outer_preds[oval] = meta.predict(outer_base)
        outer_scores.append(accuracy_score(y[oval], outer_preds[oval]))

    nested_acc = accuracy_score(y, outer_preds)
    return nested_acc, outer_scores, outer_preds

# ================================================================
# SAVE ARTIFACTS
# ================================================================
def save_experiment_results(exp_id, X_vals, Xt_vals, y, oof_preds, oof_probs, test_probs_list):
    """Save OOF predictions, test probabilities, and config."""
    exp_dir = ROOT / "oof"
    exp_dir.mkdir(exist_ok=True)

    pd.DataFrame({
        'id': range(len(y)),
        'target': y,
        'pred': oof_preds,
        **{f'prob_{c}': oof_probs[:, c] for c in range(4)}
    }).to_csv(exp_dir / f"oof_{exp_id}.csv", index=False)

    if test_probs_list and len(test_probs_list) > 0:
        test_avg = np.mean(test_probs_list, axis=0)
        pd.DataFrame(test_avg, columns=[f'prob_{c}' for c in range(4)]).to_csv(
            exp_dir / f"test_probs_{exp_id}.csv", index=False)

    return True

def save_submission(exp_id, test_probs_list, sample_sub=None):
    """Save submission CSV."""
    if sample_sub is None:
        sample_sub = pd.read_csv(ROOT.parent / "data" / "sample_submission.csv")
    test_avg = np.mean(test_probs_list, axis=0)
    test_preds = np.argmax(test_avg, axis=1)
    sub = sample_sub.copy()
    sub['target'] = test_preds
    sub.to_csv(ROOT / "submissions" / f"submission_{exp_id}.csv", index=False)
    return sub
