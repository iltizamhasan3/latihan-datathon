"""
EXP-013: Tree-guided deep interaction features.
Use CatBoost feature importance to select top features,
then generate all pairwise products/ratios among top 15.
Add interaction features to best stacking pipeline.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats as sp_stats

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
import catboost as cb

import warnings; warnings.filterwarnings('ignore')
SEED = 42; EX_ID = "EXP-013"
EXT_DIR = Path(__file__).parents[1]

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
y = train['target'].values; n = len(y)

X = get_all_features(train).fillna(0).replace([np.inf, -np.inf], 0)
Xt = get_all_features(test).fillna(0).replace([np.inf, -np.inf], 0)
for pf, cols, nc in [('week', WEEK_COLS, 12), ('act', ACTIVITY_COLS, 16)]:
    for seq_data, df in [(train[cols].fillna(0).values, X), (test[cols].fillna(0).values, Xt)]:
        xa = np.arange(nc)
        slopes = np.array([sp_stats.linregress(xa, seq_data[i])[0] if np.std(seq_data[i])>1e-10 else 0 for i in range(len(seq_data))])
        accel = np.array([np.polyfit(xa, seq_data[i], 2)[0]*2 for i in range(len(seq_data))])
        autocorr = np.array([np.corrcoef(seq_data[i,:-1], seq_data[i,1:])[0,1] if np.std(seq_data[i,:-1])>1e-10 and np.std(seq_data[i,1:])>1e-10 else 0 for i in range(len(seq_data))])
        fft = np.abs(np.fft.fft(seq_data, axis=1)); fft_p = fft[:, :nc//2]**2
        fft_n = fft_p / (fft_p.sum(axis=1, keepdims=True) + 1e-10); ent = -np.sum(fft_n * np.log(fft_n + 1e-10), axis=1)
        df[f'seq_{pf}_slope'] = slopes; df[f'seq_{pf}_accel'] = accel
        df[f'seq_{pf}_autocorr'] = autocorr; df[f'seq_{pf}_entropy'] = ent

X_vals = X.values; Xt_vals = Xt.values
print(f"Base features: {X_vals.shape[1]}")

# ================================================================
# Step 1: Get tree feature importance
# ================================================================
print("\n=== Feature Importance from CatBoost ===")
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X_vals)

cb_imp = cb.CatBoostClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, random_seed=SEED, verbose=0)
cb_imp.fit(X_scaled, y)
imp = cb_imp.get_feature_importance()
top_n = min(15, len(imp))
top_idx = np.argsort(-imp)[:top_n]
top_scores = imp[top_idx]

feat_names = X.columns.tolist() if hasattr(X, 'columns') else [f'f{i}' for i in range(X_vals.shape[1])]
print(f"  Top {top_n} features:")
for i, idx in enumerate(top_idx):
    print(f"    {i+1}. {feat_names[idx]}: {imp[idx]:.4f}")

# ================================================================
# Step 2: Generate interaction features (products among top 15)
# ================================================================
print(f"\n=== Generating Pairwise Products of Top {top_n} ===")

X_top = X_vals[:, top_idx]
Xt_top = Xt_vals[:, top_idx]

interactions = []
n_interactions = 0
for i in range(top_n):
    for j in range(i+1, top_n):
        # Product
        interactions.append(X_top[:, i] * X_top[:, j])
        n_interactions += 1
        # Ratio
        num = X_top[:, i]
        den = np.abs(X_top[:, j]) + 1e-10
        interactions.append(num / den)
        n_interactions += 1
        # Difference
        interactions.append(X_top[:, i] - X_top[:, j])
        n_interactions += 1
        # Sum
        interactions.append(X_top[:, i] + X_top[:, j])
        n_interactions += 1

# Also for test
interactions_te = []
for i in range(top_n):
    for j in range(i+1, top_n):
        interactions_te.append(Xt_top[:, i] * Xt_top[:, j])
        interactions_te.append(Xt_top[:, i] / (np.abs(Xt_top[:, j]) + 1e-10))
        interactions_te.append(Xt_top[:, i] - Xt_top[:, j])
        interactions_te.append(Xt_top[:, i] + Xt_top[:, j])

X_int = np.column_stack(interactions)
Xt_int = np.column_stack(interactions_te)
print(f"  Interaction features: {X_int.shape[1]}")

# Handle inf/nan
X_int = np.nan_to_num(X_int, nan=0.0, posinf=0.0, neginf=0.0)
Xt_int = np.nan_to_num(Xt_int, nan=0.0, posinf=0.0, neginf=0.0)

# ================================================================
# Step 3: Stacking with interactions
# ================================================================
print("\n=== Stacking with Interaction Features ===")

X_full = np.column_stack([X_vals, X_int])
Xt_full = np.column_stack([Xt_vals, Xt_int])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof = np.zeros(n, dtype=int); folds = []; test_probs = []

t0 = time.time()
for fi, (tr, val) in enumerate(cv.split(X_full, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_full[tr]); X_val = scaler.transform(X_full[val]); X_te = scaler.transform(Xt_full)
    rs = 42 + fi

    svc = SVC(C=10, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
    cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0)
    cb_m.fit(X_tr, y[tr])

    meta_val = np.column_stack([svc.predict_proba(X_val), cb_m.predict_proba(X_val)])
    meta_te = np.column_stack([svc.predict_proba(X_te), cb_m.predict_proba(X_te)])

    meta = LogisticRegression(C=0.3, max_iter=2000, random_state=rs)
    meta.fit(meta_val, y[val])
    oof[val] = meta.predict(meta_val)
    folds.append(accuracy_score(y[val], oof[val]))
    test_probs.append(meta.predict_proba(meta_te))

acc = accuracy_score(y, oof)
f1 = f1_score(y, oof, average='macro')
elapsed = time.time() - t0

# ================================================================
# Step 4: Keep only best interactions via MI
# ================================================================
print("\n=== Filtering Interactions by MI ===")
from sklearn.feature_selection import mutual_info_classif
mi_vals = mutual_info_classif(X_int, y, random_state=SEED)
top_mi_idx = np.argsort(-mi_vals)[:100]
X_int_best = X_int[:, top_mi_idx]
Xt_int_best = Xt_int[:, top_mi_idx]
print(f"  Selected top 100 interactions by MI")

X_full_best = np.column_stack([X_vals, X_int_best])
Xt_full_best = np.column_stack([Xt_vals, Xt_int_best])

cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof2 = np.zeros(n, dtype=int); folds2 = []

for fi, (tr, val) in enumerate(cv2.split(X_full_best, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_full_best[tr]); X_val = scaler.transform(X_full_best[val])
    rs = 42 + fi
    svc = SVC(C=10, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
    cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0)
    cb_m.fit(X_tr, y[tr])
    meta_val = np.column_stack([svc.predict_proba(X_val), cb_m.predict_proba(X_val)])
    meta = LogisticRegression(C=0.3, max_iter=2000, random_state=rs)
    meta.fit(meta_val, y[val])
    oof2[val] = meta.predict(meta_val)
    folds2.append(accuracy_score(y[val], oof2[val]))

acc2 = accuracy_score(y, oof2)
print(f"\n  With MI-filtered interactions: {acc2:.4f}")

# ================================================================
# RESULTS
# ================================================================
print(f"\n{'='*50}")
print(f"EXP-013 INTERACTION RESULTS")
print(f"{'='*50}")
print(f"  All interactions ({X_int.shape[1]}): {acc:.4f}")
print(f"  MI-filtered (100): {acc2:.4f}")
print(f"  Current best:       0.5988")
best_acc = max(acc, acc2)
delta = best_acc - 0.5988
print(f"  Best vs 0.5988:     {delta:+.4f}")

# Save
import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID, 'parent_id': 'EXP-012',
    'hypothesis': 'Tree-guided interaction features (pairwise products/ratios among top 15 most important)',
    'feature_family': f'base+sequence+{X_int.shape[1]}_interactions',
    'model_family': 'Stack_SVC(C=10)+CB(d=4)_LR(C=0.3)',
    'parameters': json.dumps({'top_n_features': top_n, 'interactions': X_int.shape[1], 'mi_filtered': 100}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in (folds2 if acc2 > acc else folds)]),
    'mean_accuracy': float(best_acc),
    'macro_f1': float(f1 if best_acc == acc else 0),
    'minimum_fold': float(min(folds2 if acc2 > acc else folds)),
    'accepted': best_acc > 0.5988,
    'rejection_reason': '' if best_acc > 0.5988 else 'plateau confirmed',
    'next_hypothesis': 'Synthesis — extreme search complete at 14 experiments across 17 stages',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists: writer.writeheader()
    writer.writerow(row)
print(f"Done.")
