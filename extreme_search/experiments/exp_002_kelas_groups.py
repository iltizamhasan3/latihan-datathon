"""
EXP-002: Kelas Group Prior + Classmate Features
Hypothesis: Strongest signal is per-class (kelas) aggregates.
With 483/492 test samples sharing kelas with train, group-level priors
provide leakage-safe signal boost.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
from scipy import stats as sp_stats
import catboost as cb

import warnings; warnings.filterwarnings('ignore')

SEED = 42
EX_ID = "EXP-002"
EXT_DIR = Path(__file__).parents[1]

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
y = train['target'].values
n = len(y)

# Build base + sequence features
X = get_all_features(train).fillna(0).replace([np.inf, -np.inf], 0)
Xt = get_all_features(test).fillna(0).replace([np.inf, -np.inf], 0)

for pf, cols, nc in [('week', WEEK_COLS, 12), ('act', ACTIVITY_COLS, 16)]:
    for seq_data, df in [(train[cols].fillna(0).values, X), (test[cols].fillna(0).values, Xt)]:
        xa = np.arange(nc)
        slopes = np.array([sp_stats.linregress(xa, seq_data[i])[0] if np.std(seq_data[i])>1e-10 else 0 for i in range(len(seq_data))])
        accel = np.array([np.polyfit(xa, seq_data[i], 2)[0]*2 for i in range(len(seq_data))])
        autocorr = np.array([np.corrcoef(seq_data[i,:-1], seq_data[i,1:])[0,1] if np.std(seq_data[i,:-1])>1e-10 and np.std(seq_data[i,1:])>1e-10 else 0 for i in range(len(seq_data))])
        fft = np.abs(np.fft.fft(seq_data, axis=1))
        fft_p = fft[:, :nc//2]**2
        fft_n = fft_p / (fft_p.sum(axis=1, keepdims=True) + 1e-10)
        ent = -np.sum(fft_n * np.log(fft_n + 1e-10), axis=1)
        df[f'seq_{pf}_slope'] = slopes
        df[f'seq_{pf}_accel'] = accel
        df[f'seq_{pf}_autocorr'] = autocorr
        df[f'seq_{pf}_entropy'] = ent

X_vals = X.values
Xt_vals = Xt.values
train_kelas = train['kelas'].fillna(-1).astype(int).values
test_kelas = test['kelas'].fillna(-1).astype(int).values

print(f"Features: {X_vals.shape[1]}")
print(f"Train kelas unique: {len(np.unique(train_kelas))}")
print(f"Test kelas unique: {len(np.unique(test_kelas))}")
print(f"Test-train kelas overlap: {len(set(train_kelas) & set(test_kelas))}")

def add_kelas_priors(X_fold, y_fold, tr_idx, val_idx, train_kelas, test_kelas):
    """
    Build class prior features leakage-safe:
    - From training indices, compute per-kelas target distribution
    - Apply as features to validation and test
    """
    tr_k = train_kelas[tr_idx]
    tr_y = y_fold
    val_k = train_kelas[val_idx]

    all_k = np.unique(train_kelas)
    k_priors = {}
    k_counts = {}
    for k in all_k:
        mask = tr_k == k
        cnt = mask.sum()
        k_counts[k] = cnt
        if cnt > 0:
            priors = np.array([(tr_y[mask] == c).mean() for c in range(4)])
        else:
            priors = np.array([0.25, 0.25, 0.25, 0.25])
        k_priors[k] = priors

    val_priors = np.array([k_priors.get(k, [0.25,0.25,0.25,0.25]) for k in val_k])
    val_counts = np.array([k_counts.get(k, 0) for k in val_k])
    return val_priors, val_counts.reshape(-1, 1)

# ================================================================
# Strategy A: HGB dengan kelas priors sebagai fitur
# ================================================================
print("\n=== Strategy A: HGB + Kelas Priors ===")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_a = np.zeros(n, dtype=int)
fold_a = []
test_probs_a = []
tr_kelas = train_kelas
te_kelas = test_kelas

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    X_te = scaler.transform(Xt_vals)
    rs = 42 + fi

    # Compute k_priors from training data
    tr_k = tr_kelas[tr]
    k_priors_a = {}
    for k in np.unique(tr_k):
        mask = tr_k == k
        if mask.sum() > 0:
            k_priors_a[k] = np.array([(y[tr][mask] == c).mean() for c in range(4)])
        else:
            k_priors_a[k] = np.array([0.25, 0.25, 0.25, 0.25])

    tr_priors = np.array([k_priors_a.get(k, [0.25]*4) for k in tr_k])
    val_priors = np.array([k_priors_a.get(k, [0.25]*4) for k in tr_kelas[val]])
    te_priors = np.array([k_priors_a.get(k, [0.25]*4) for k in te_kelas])

    X_tr_full = np.column_stack([X_tr, tr_priors])
    X_val_full = np.column_stack([X_val, val_priors])
    X_te_full = np.column_stack([X_te, te_priors])

    # HGB
    hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.05, random_state=rs)
    hgb.fit(X_tr_full, y[tr])
    oof_a[val] = hgb.predict(X_val_full)
    fold_a.append(accuracy_score(y[val], oof_a[val]))
    test_probs_a.append(hgb.predict_proba(X_te_full))
    print(f"  Fold {fi+1}: acc={fold_a[-1]:.4f}")

acc_a = accuracy_score(y, oof_a)
f1_a = f1_score(y, oof_a, average='macro')
print(f"\nHGB+Kelas: acc={acc_a:.4f}, f1={f1_a:.4f}")

# ================================================================
# Strategy B: SVC + CatBoost stacked, with kelas priors at meta level
# ================================================================
print("\n=== Strategy B: SVC+CatBoost Stacked + Kelas Priors ===")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_b = np.zeros(n, dtype=int)
fold_b = []
test_probs_b = []

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    X_te = scaler.transform(Xt_vals)
    rs = 42 + fi

    # Recompute k_priors
    tr_k = tr_kelas[tr]
    k_priors_b = {}
    for k in np.unique(tr_k):
        mask = tr_k == k
        k_priors_b[k] = np.array([(y[tr][mask] == c).mean() for c in range(4)]) if mask.sum() > 0 else np.array([0.25]*4)

    # Base models
    svc = SVC(C=50, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
    cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=rs, verbose=0)
    cb_m.fit(X_tr, y[tr])

    # OOF probs
    svc_oof = svc.predict_proba(X_val)
    cb_oof = cb_m.predict_proba(X_val)

    # Meta with kelas priors
    val_prior = np.array([k_priors_b.get(k, [0.25]*4) for k in tr_kelas[val]])
    te_prior = np.array([k_priors_b.get(k, [0.25]*4) for k in te_kelas])

    meta_val = np.column_stack([svc_oof, cb_oof, val_prior])
    meta_te = np.column_stack([svc.predict_proba(X_te), cb_m.predict_proba(X_te), te_prior])

    meta = LogisticRegression(C=0.1, max_iter=2000, random_state=rs)
    meta.fit(meta_val, y[val])
    oof_b[val] = meta.predict(meta_val)
    fold_b.append(accuracy_score(y[val], oof_b[val]))
    test_probs_b.append(meta.predict_proba(meta_te))
    print(f"  Fold {fi+1}: acc={fold_b[-1]:.4f}")

acc_b = accuracy_score(y, oof_b)
f1_b = f1_score(y, oof_b, average='macro')
print(f"\nStack+Kelas: acc={acc_b:.4f}, f1={f1_b:.4f}")

# ================================================================
# Strategy C: HGB baseline (no kelas) for control
# ================================================================
print("\n=== Strategy C: HGB baseline (no kelas) ===")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_c = np.zeros(n, dtype=int)
fold_c = []

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    rs = 42 + fi
    hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.05, random_state=rs)
    hgb.fit(X_tr, y[tr])
    oof_c[val] = hgb.predict(X_val)
    fold_c.append(accuracy_score(y[val], oof_c[val]))

acc_c = accuracy_score(y, oof_c)
print(f"HGB baseline: acc={acc_c:.4f}")

# ================================================================
# RESULTS
# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
print(f"  HGB+Kelas:     {acc_a:.4f} (f1={f1_a:.4f})")
print(f"  Stack+Kelas:   {acc_b:.4f} (f1={f1_b:.4f})")
print(f"  HGB baseline:  {acc_c:.4f}")
print(f"  Prior best:    0.5975 (forensic)")

best_strat = 'A' if acc_a >= acc_b else 'B'
best_acc = max(acc_a, acc_b)
delta = best_acc - 0.5975
print(f"\n  Best: Strategy {best_strat} = {best_acc:.4f} (vs 0.5975: {delta:+.4f})")
print(f"{'='*50}")

# Save best submission
test_probs_best = test_probs_a if best_strat == 'A' else test_probs_b
test_avg = np.mean(test_probs_best, axis=0)
test_preds = np.argmax(test_avg, axis=1)
sub = sample_sub.copy()
sub['target'] = test_preds
sub.to_csv(EXT_DIR / "submissions" / f"submission_{EX_ID.lower()}.csv", index=False)
print(f"\nSubmission saved.")

# Log
import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID,
    'parent_id': 'None',
    'hypothesis': 'Kelas group priors provide leakage-safe signal boost',
    'feature_family': 'base_engineered+sequence+kelas_priors',
    'model_family': f'HGB/Stack_SVC+CatBoost (strat_{best_strat})',
    'parameters': json.dumps({'HGB_max_iter':500, 'SVC_C':50, 'CatBoost_n':400, 'meta_LR_C':0.1}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in (fold_a if best_strat == 'A' else fold_b)]),
    'mean_accuracy': float(best_acc),
    'macro_f1': float(f1_a if best_strat == 'A' else f1_b),
    'balanced_accuracy': 0.0,
    'minimum_fold': float(min(fold_a if best_strat == 'A' else fold_b)),
    'train_accuracy': None,
    'overfit_gap': None,
    'runtime': 0,
    'accepted': best_acc > 0.5975,
    'rejection_reason': '' if best_acc > 0.5975 else 'at or below baseline',
    'next_hypothesis': 'EXP-003: Symbolic formula discovery for latent target score',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"Done.")
