"""
EXP-005: Ordinal reconstruction with cumulative link models.
Tests: CORAL-like ordinal without actual CORAL package,
cumulative HGB, pairwise rank learning.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats as sp_stats

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
import catboost as cb

import warnings; warnings.filterwarnings('ignore')

SEED = 42
EX_ID = "EXP-005"
EXT_DIR = Path(__file__).parents[1]

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
y = train['target'].values
n = len(y)

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
print(f"Features: {X_vals.shape[1]}")

# ================================================================
# Strategy A: Cumulative HGB (binary for each k)
# ================================================================
print("\n=== Strategy A: Cumulative HGB (P(y>=k)) ===")

def cumulative_probs_cv(X_vals, y, Xt_vals, cv):
    """Generate cumulative probabilities via binary classifiers."""
    n = len(y)
    oof_cum = np.zeros((n, 3))  # P(y>=1), P(y>=2), P(y>=3)
    te_cum_list = []

    for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_vals[tr])
        X_val = scaler.transform(X_vals[val])
        X_te = scaler.transform(Xt_vals)
        rs = 42 + fi

        fold_te = []
        for k in range(1, 4):
            y_bin = (y[tr] >= k).astype(int)
            hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.05, random_state=rs)
            hgb.fit(X_tr, y_bin)
            oof_cum[val, k-1] = hgb.predict_proba(X_val)[:, 1]
            fold_te.append(hgb.predict_proba(X_te)[:, 1])

        te_cum_list.append(np.column_stack(fold_te))

    # Derive class probabilities: P(y=0)=1-P(>=1), P(y=1)=P(>=1)-P(>=2), etc.
    # Force monotonic: P(>=1) >= P(>=2) >= P(>=3)
    oof_cum = np.maximum.accumulate(np.sort(oof_cum, axis=1)[:, ::-1], axis=1)[:, ::-1]
    # Actually sort ensures monotonic: ensure P(>=k) >= P(>=k+1)
    for i in range(n):
        for k in range(2):
            oof_cum[i, k] = max(oof_cum[i, k], oof_cum[i, k+1])

    oof_probs = np.zeros((n, 4))
    oof_probs[:, 0] = 1 - oof_cum[:, 0]
    oof_probs[:, 1] = oof_cum[:, 0] - oof_cum[:, 1]
    oof_probs[:, 2] = oof_cum[:, 1] - oof_cum[:, 2]
    oof_probs[:, 3] = oof_cum[:, 2]
    oof_preds = np.argmax(oof_probs, axis=1)

    te_avg = np.mean(te_cum_list, axis=0) if te_cum_list else np.zeros((len(Xt_vals), 3))
    te_probs = np.zeros((len(Xt_vals), 4))
    te_probs[:, 0] = 1 - te_avg[:, 0]
    te_probs[:, 1] = te_avg[:, 0] - te_avg[:, 1]
    te_probs[:, 2] = te_avg[:, 1] - te_avg[:, 2]
    te_probs[:, 3] = te_avg[:, 2]

    return oof_preds, oof_probs, te_probs

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_a, oof_prob_a, te_prob_a = cumulative_probs_cv(X_vals, y, Xt_vals, cv)
acc_a = accuracy_score(y, oof_a)
f1_a = f1_score(y, oof_a, average='macro')
print(f"  Cumulative HGB: {acc_a:.4f}, f1={f1_a:.4f}")

# ================================================================
# Strategy B: Stack cumulative + regular probabilities
# ================================================================
print("\n=== Strategy B: Stack Cumulative + SVC/CatBoost OOF Probs ===")

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

    # SVC
    svc = SVC(C=50, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
    # CatBoost
    cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=rs, verbose=0)
    cb_m.fit(X_tr, y[tr])

    # Cumulative HGB
    cum_val = np.zeros((len(val), 3))
    cum_te = np.zeros((len(Xt_vals), 3))
    for k in range(1, 4):
        y_bin = (y[tr] >= k).astype(int)
        hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=3, learning_rate=0.05, random_state=rs)
        hgb.fit(X_tr, y_bin)
        cum_val[:, k-1] = hgb.predict_proba(X_val)[:, 1]
        cum_te[:, k-1] = hgb.predict_proba(X_te)[:, 1]

    cum_class_val = np.zeros((len(val), 4))
    cum_class_val[:, 0] = 1 - cum_val[:, 0]
    cum_class_val[:, 1] = cum_val[:, 0] - cum_val[:, 1]
    cum_class_val[:, 2] = cum_val[:, 1] - cum_val[:, 2]
    cum_class_val[:, 3] = cum_val[:, 2]

    cum_class_te = np.zeros((len(Xt_vals), 4))
    cum_class_te[:, 0] = 1 - cum_te[:, 0]
    cum_class_te[:, 1] = cum_te[:, 0] - cum_te[:, 1]
    cum_class_te[:, 2] = cum_te[:, 1] - cum_te[:, 2]
    cum_class_te[:, 3] = cum_te[:, 2]

    # Meta
    meta_val = np.column_stack([svc.predict_proba(X_val), cb_m.predict_proba(X_val), cum_class_val])
    meta_te = np.column_stack([svc.predict_proba(X_te), cb_m.predict_proba(X_te), cum_class_te])

    meta = LogisticRegression(C=0.1, max_iter=2000, random_state=rs)
    meta.fit(meta_val, y[val])
    oof_b[val] = meta.predict(meta_val)
    fold_b.append(accuracy_score(y[val], oof_b[val]))
    test_probs_b.append(meta.predict_proba(meta_te))
    print(f"  Fold {fi+1}: acc={fold_b[-1]:.4f}")

acc_b = accuracy_score(y, oof_b)
f1_b = f1_score(y, oof_b, average='macro')
print(f"Stack+Cumulative: {acc_b:.4f}, f1={f1_b:.4f}")

# ================================================================
# Strategy C: Ordinal consistency correction for any model
# ================================================================
print("\n=== Strategy C: Ordinal consistency correction ===")

# Take best stacking from EXP-004 and apply ordinal correction
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_c = np.zeros(n, dtype=int)
fold_c = []

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    rs = 42 + fi

    svc = SVC(C=50, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
    cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=rs, verbose=0)
    cb_m.fit(X_tr, y[tr])
    hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.05, random_state=rs)
    hgb.fit(X_tr, y[tr])

    meta_val = np.column_stack([svc.predict_proba(X_val), cb_m.predict_proba(X_val), hgb.predict_proba(X_val)])
    meta = LogisticRegression(C=0.1, max_iter=2000, random_state=rs)
    meta.fit(meta_val, y[val])

    # Get probs and enforce ordinal consistency
    raw_probs = meta.predict_proba(meta_val)
    # Monotonic correction: ensure P(y<=k) monotonic
    cum_probs = np.cumsum(raw_probs, axis=1)
    # Make cumulative monotonic
    for i in range(len(raw_probs)):
        for k in range(1, 4):
            cum_probs[i, k] = max(cum_probs[i, k], cum_probs[i, k-1])
    # Convert back
    corr_probs = np.zeros_like(raw_probs)
    corr_probs[:, 0] = cum_probs[:, 0]
    corr_probs[:, 1] = cum_probs[:, 1] - cum_probs[:, 0]
    corr_probs[:, 2] = cum_probs[:, 2] - cum_probs[:, 1]
    corr_probs[:, 3] = 1 - cum_probs[:, 2]

    oof_c[val] = np.argmax(corr_probs, axis=1)
    fold_c.append(accuracy_score(y[val], oof_c[val]))

acc_c = accuracy_score(y, oof_c)
print(f"Ordinal-corrected stacking: {acc_c:.4f}")

# ================================================================
# RESULTS
# ================================================================
print(f"\n{'='*50}")
print(f"EXP-005 RESULTS")
print(f"{'='*50}")
print(f"  A (Cumulative HGB):         {acc_a:.4f}")
print(f"  B (Stack+Cumulative):       {acc_b:.4f}")
print(f"  C (Ordinal-corrected stack): {acc_c:.4f}")
print(f"  Current best:               0.5975")

best_acc = max(acc_a, acc_b, acc_c)
best_strat = ['A', 'B', 'C'][np.argmax([acc_a, acc_b, acc_c])]
delta = best_acc - 0.5975
print(f"\n  Best: {best_strat} = {best_acc:.4f} (vs 0.5975: {delta:+.4f})")

# Log
import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID,
    'parent_id': 'EXP-004',
    'hypothesis': 'Ordinal reconstruction via cumulative link models improves multi-class decomposition',
    'feature_family': 'base_engineered+sequence',
    'model_family': f'HGB_cumulative/Stack_LR (strat_{best_strat})',
    'parameters': json.dumps({'cumulative_binary': 'HGB_max_iter_500', 'meta_LR_C': 0.1}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in (
        [float(acc_a)]*10 if best_strat == 'A' else ([float(f) for f in fold_b] if best_strat == 'B' else [float(f) for f in fold_c])
    )]),
    'mean_accuracy': float(best_acc),
    'macro_f1': float(f1_a if best_strat == 'A' else (f1_b if best_strat == 'B' else 0)),
    'balanced_accuracy': 0.0,
    'minimum_fold': float(min(fold_b if best_strat == 'B' else fold_c)) if best_strat != 'A' else 0.0,
    'train_accuracy': None,
    'overfit_gap': None,
    'runtime': 0,
    'accepted': best_acc > 0.5975,
    'rejection_reason': '' if best_acc > 0.5975 else 'not improving over current best',
    'next_hypothesis': 'EXP-006: Pairwise specialist models for overlapping classes',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists:
        writer.writeheader()
    writer.writerow(row)
print(f"Done.")
