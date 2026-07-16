"""
EXP-003: Symbolic formula search for latent target score.
Discovers interpretable formulas: target = bin(f(w1*x1,...,wn*xn), thresholds)
Uses genetic programming to search expression space.
"""
import numpy as np, pandas as pd, sys, json, time, random
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, r2_score
from scipy import stats

import warnings; warnings.filterwarnings('ignore')

SEED = 42
EX_ID = "EXP-003"
EXT_DIR = Path(__file__).parents[1]
np.random.seed(SEED)
random.seed(SEED)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
y = train['target'].values
n = len(y)

# Use raw features directly for formula search (need interpretability)
feat_cols = [c for c in train.columns if c not in ['id', 'target']]
X_raw = train[feat_cols].fillna(0).values
Xt_raw = test[feat_cols].fillna(0).values
feature_names = feat_cols

print(f"Raw features: {X_raw.shape[1]}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
Xt_scaled = scaler.transform(Xt_raw)

# ================================================================
# Strategy A: Genetic formula search via random combinations
# Search: w1*f1 + w2*f2 + ... + nonlinear transforms
# ================================================================
print("\n=== Strategy A: Formula Search ===")

# Pre-compute candidate features (unary transforms of top features)
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X_scaled, y, random_state=SEED)
top_feat_idx = np.argsort(-mi)[:30]

candidates = []
cand_names = []

# Original top features
for idx in top_feat_idx:
    candidates.append(X_scaled[:, idx])
    cand_names.append(f'f{idx}_{feature_names[idx]}')

# Square, sqrt, log, abs of top features
for idx in top_feat_idx:
    v = X_scaled[:, idx]
    # Square
    candidates.append(v**2)
    cand_names.append(f'f{idx}_sq')
    # Absolute
    candidates.append(np.abs(v))
    cand_names.append(f'f{idx}_abs')
    # Clip (ReLU-like)
    candidates.append(np.maximum(v, 0))
    cand_names.append(f'f{idx}_relu')
    candidates.append(np.minimum(v, 0))
    cand_names.append(f'f{idx}_neg')

candidates = np.column_stack(candidates)
print(f"Candidate features: {candidates.shape[1]}")

# Greedy forward selection + Ridge regression
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
fold_scores = []
oof_fs = np.zeros(n, dtype=int)

for fi, (tr, val) in enumerate(cv.split(X_scaled, y)):
    tr_c = candidates[tr]
    val_c = candidates[val]
    rs = 42 + fi

    # Greedy feature selection: add one feature at a time
    selected = []
    remaining = list(range(candidates.shape[1]))
    best_score = -np.inf

    for _ in range(min(20, len(remaining))):
        best_j = -1
        for j in remaining:
            trial = selected + [j]
            model = Ridge(alpha=1.0, random_state=rs)
            model.fit(tr_c[:, trial], y[tr])
            # Score on train
            tr_pred = model.predict(tr_c[:, trial])
            # Find optimal thresholds
            best_tr = 0
            for t1 in np.percentile(tr_pred, [20, 25, 30, 33]):
                for t2 in np.percentile(tr_pred, [45, 50, 55, 60]):
                    for t3 in np.percentile(tr_pred, [70, 75, 80, 85]):
                        if t1 < t2 < t3:
                            p = np.digitize(tr_pred, [t1, t2, t3])
                            s = (p == y[tr]).mean()
                            if s > best_tr:
                                best_tr = s
            if best_tr > best_score:
                best_score = best_tr
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
            remaining.remove(best_j)
        else:
            break

    # Final model
    model = Ridge(alpha=1.0, random_state=rs)
    model.fit(tr_c[:, selected], y[tr])
    val_pred = model.predict(val_c[:, selected])

    # Find thresholds on training predictions
    tr_pred = model.predict(tr_c[:, selected])
    best_t = None
    best_v = 0
    for t1 in np.percentile(tr_pred, [20, 25, 30, 33]):
        for t2 in np.percentile(tr_pred, [45, 50, 55, 60]):
            for t3 in np.percentile(tr_pred, [70, 75, 80, 85]):
                if t1 < t2 < t3:
                    p = np.digitize(tr_pred, [t1, t2, t3])
                    s = (p == y[tr]).mean()
                    if s > best_v:
                        best_v = s
                        best_t = (t1, t2, t3)

    if best_t:
        oof_fs[val] = np.digitize(val_pred, best_t)
    fa = accuracy_score(y[val], oof_fs[val])
    fold_scores.append(fa)

    # Show selected features
    selected_names = [cand_names[j] for j in selected[:5]]
    print(f"  Fold {fi+1}: acc={fa:.4f}, selected={selected_names}")

acc_fs = accuracy_score(y, oof_fs)
print(f"\nFormula search OOF: {acc_fs:.4f}")

# ================================================================
# Strategy B: Sparse ratio features
# Search for informative feature ratios
# ================================================================
print("\n=== Strategy B: Feature Ratios ===")

# Generate pairwise ratios from top features
ratio_names = []
ratio_vals = []
for i in range(30):
    for j in range(i+1, 30):
        fi_v = np.abs(X_scaled[:, top_feat_idx[i]]) + 1e-10
        fj_v = np.abs(X_scaled[:, top_feat_idx[j]]) + 1e-10
        ratio_vals.append(fi_v / fj_v)
        ratio_names.append(f'ratio_{feature_names[top_feat_idx[i]]}_{feature_names[top_feat_idx[j]]}')

X_ratio = np.column_stack(ratio_vals)
print(f"Ratio features: {X_ratio.shape[1]}")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
fold_r = []
oof_r = np.zeros(n, dtype=int)

for fi, (tr, val) in enumerate(cv.split(X_scaled, y)):
    # Select top ratio features by MI on training fold
    mi_r = mutual_info_classif(X_ratio[tr], y[tr], random_state=SEED)
    top_r = np.argsort(-mi_r)[:30]

    rs = 42 + fi
    hgb = Ridge(alpha=0.5, random_state=rs)
    hgb.fit(X_ratio[tr][:, top_r], y[tr])
    val_p = hgb.predict(X_ratio[val][:, top_r])

    # Thresholds
    tr_p = hgb.predict(X_ratio[tr][:, top_r])
    best_t = None
    best_v = 0
    for t1 in np.percentile(tr_p, [20, 25, 30, 33]):
        for t2 in np.percentile(tr_p, [45, 50, 55, 60]):
            for t3 in np.percentile(tr_p, [70, 75, 80, 85]):
                if t1 < t2 < t3:
                    p = np.digitize(tr_p, [t1, t2, t3])
                    s = (p == y[tr]).mean()
                    if s > best_v:
                        best_v = s
                        best_t = (t1, t2, t3)

    if best_t:
        oof_r[val] = np.digitize(val_p, best_t)
    fold_r.append(accuracy_score(y[val], oof_r[val]))
    # Show top 3 ratio features
    top_r_names = [ratio_names[top_r[k]] for k in range(min(3, len(top_r)))]
    print(f"  Fold {fi+1}: acc={fold_r[-1]:.4f}, top={top_r_names}")

acc_r = accuracy_score(y, oof_r)
print(f"Ratio search OOF: {acc_r:.4f}")

# ================================================================
# Strategy C: SVC with formula-based features
# ================================================================
print("\n=== Strategy C: Optimal formula via ElasticNet ===")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
fold_c = []
oof_c = np.zeros(n, dtype=int)
test_probs_c = []

for fi, (tr, val) in enumerate(cv.split(X_scaled, y)):
    # ElasticNet to find sparse weighted combination
    en = ElasticNet(alpha=0.005, l1_ratio=0.3, max_iter=10000, random_state=42+fi, selection='random')
    from sklearn.linear_model import ElasticNet
    en.fit(X_scaled[tr], y[tr])
    tr_score = en.predict(X_scaled[tr])
    val_score = en.predict(X_scaled[val])

    # Thresholds
    best_t = None
    best_v = 0
    for t1 in np.percentile(tr_score, [20, 25, 30, 33]):
        for t2 in np.percentile(tr_score, [45, 50, 55, 60]):
            for t3 in np.percentile(tr_score, [70, 75, 80, 85]):
                if t1 < t2 < t3:
                    p = np.digitize(tr_score, [t1, t2, t3])
                    s = (p == y[tr]).mean()
                    if s > best_v:
                        best_v = s
                        best_t = (t1, t2, t3)

    if best_t:
        oof_c[val] = np.digitize(val_score, best_t)
    fold_c.append(accuracy_score(y[val], oof_c[val]))
    print(f"  Fold {fi+1}: acc={fold_c[-1]:.4f}, nonzero_coefs={(np.abs(en.coef_) > 1e-6).sum()}")

acc_c = accuracy_score(y, oof_c)
print(f"ElasticNet formula OOF: {acc_c:.4f}")

# ================================================================
# RESULTS
# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS - EXP-003 Formula Search")
print(f"{'='*50}")
print(f"  A (Formula search):      {acc_fs:.4f}")
print(f"  B (Feature ratios):      {acc_r:.4f}")
print(f"  C (ElasticNet formula):  {acc_c:.4f}")
best_acc = max(acc_fs, acc_r, acc_c)
best_strat = ['A', 'B', 'C'][np.argmax([acc_fs, acc_r, acc_c])]
delta = best_acc - 0.5975
print(f"\n  Best: {best_strat} = {best_acc:.4f} (vs 0.5975: {delta:+.4f})")

# Log
import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID,
    'parent_id': 'EXP-001',
    'hypothesis': 'Target = bin(formula(weighted_features), thresholds). Symbolic formula search with Ridge/ElasticNet',
    'feature_family': f'raw_features+ratios (strat_{best_strat})',
    'model_family': 'Ridge/ElasticNet + threshold binning',
    'parameters': json.dumps({'strategy': best_strat, 'feature_selection': 'greedy/MI'}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in (
        fold_scores if best_strat == 'A' else (fold_r if best_strat == 'B' else fold_c)
    )]),
    'mean_accuracy': float(best_acc),
    'macro_f1': 0.0,
    'balanced_accuracy': 0.0,
    'minimum_fold': float(min(
        fold_scores if best_strat == 'A' else (fold_r if best_strat == 'B' else fold_c)
    )),
    'train_accuracy': None,
    'overfit_gap': None,
    'runtime': 0,
    'accepted': best_acc > 0.5975,
    'rejection_reason': '' if best_acc > 0.5975 else 'below current best',
    'next_hypothesis': 'EXP-004: Advanced sequence + cross-sequence features for SVC stacking',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists:
        writer.writeheader()
    writer.writerow(row)
print(f"\nDone.")
