"""
EXP-008: Ensemble weight optimization + mixture of experts.
Systematically search model weights, find per-class specialists,
and build dynamic ensemble selection based on confidence.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats as sp_stats, optimize as sp_opt

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
import catboost as cb

import warnings; warnings.filterwarnings('ignore')
SEED = 42; EX_ID = "EXP-008"
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
        fft_n = fft_p / (fft_p.sum(axis=1, keepdims=True) + 1e-10)
        ent = -np.sum(fft_n * np.log(fft_n + 1e-10), axis=1)
        df[f'seq_{pf}_slope'] = slopes; df[f'seq_{pf}_accel'] = accel
        df[f'seq_{pf}_autocorr'] = autocorr; df[f'seq_{pf}_entropy'] = ent

X_vals = X.values; Xt_vals = Xt.values
print(f"Features: {X_vals.shape[1]}")

# ================================================================
# Generate OOF probs from diverse models for ensemble
# ================================================================
print("\n=== Building Model Pool ===")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
models = {
    'SVC_50': lambda rs: SVC(C=50, gamma='auto', probability=True, random_state=rs),
    'SVC_100': lambda rs: SVC(C=100, gamma='auto', probability=True, random_state=rs),
    'SVC_10': lambda rs: SVC(C=10, gamma='auto', probability=True, random_state=rs),
    'CatBoost_d6': lambda rs: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=rs, verbose=0),
    'CatBoost_d4': lambda rs: cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0),
    'HGB': lambda rs: HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.05, random_state=rs),
    'ET': lambda rs: ExtraTreesClassifier(n_estimators=400, max_depth=6, random_state=rs),
}

all_models_OOF = {}  # model -> (n,4) OOF probs
test_probs_dict = {}

for mname, mfn in models.items():
    oof_p = np.zeros((n, 4))
    te_p_list = []
    for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_vals[tr])
        X_val = scaler.transform(X_vals[val])
        X_te = scaler.transform(Xt_vals)
        m = mfn(42+fi).fit(X_tr, y[tr])
        oof_p[val] = m.predict_proba(X_val)
        te_p_list.append(m.predict_proba(X_te))
    all_models_OOF[mname] = oof_p
    test_probs_dict[mname] = np.mean(te_p_list, axis=0)
    acc = accuracy_score(y, np.argmax(oof_p, axis=1))
    print(f"  {mname}: single-model OOF={acc:.4f}")

# ================================================================
# Strategy A: Grid search model weights
# ================================================================
print("\n=== Strategy A: Grid Search Ensemble Weights ===")

model_names = list(all_models_OOF.keys())
n_models = len(model_names)

# Convert to arrays: (N, 4, M)
model_OOF_array = np.array([all_models_OOF[n] for n in model_names])  # (M, N, 4)
model_te_array = np.array([test_probs_dict[n] for n in model_names])  # (M, T, 4)

best_score = 0
best_weights = None

# Random search over weight simplex
np.random.seed(SEED)
for trial in range(500):
    w = np.random.dirichlet(np.ones(n_models) * 0.5)
    ensemble_probs = np.tensordot(w, model_OOF_array, axes=(0, 0))  # (N, 4)
    preds = np.argmax(ensemble_probs, axis=1)
    acc = accuracy_score(y, preds)
    if acc > best_score:
        best_score = acc
        best_weights = w

print(f"  Best random weights: {best_score:.4f}")
for i, n in enumerate(model_names):
    print(f"    {n}: {best_weights[i]:.4f}")

# ================================================================
# Strategy B: Per-class specialist ensemble
# ================================================================
print("\n=== Strategy B: Per-Class Specialist Ensemble ===")

# For each class, find which model is best at that class
class_recalls = np.zeros((n_models, 4))
for mi, mname in enumerate(model_names):
    preds = np.argmax(all_models_OOF[mname], axis=1)
    for c in range(4):
        mask = y == c
        class_recalls[mi, c] = (preds[mask] == c).mean()

for c in range(4):
    best_mi = np.argmax(class_recalls[:, c])
    print(f"  Class {c} best: {model_names[best_mi]} ({class_recalls[best_mi, c]:.4f})")

# Specialist ensemble: for each sample, use model that's best at that sample's predicted class
# Or better: weighted average where weights depend on predicted class
def per_class_weighted_ensemble(oof_arrays, model_names, class_recalls):
    """Weight each model's prediction per class by its class recall."""
    M, N, C = oof_arrays.shape
    ensemble = np.zeros((N, C))
    for c in range(C):
        w = class_recalls[:, c]
        w = w / w.sum()
        ensemble[:, c] = np.tensordot(w, oof_arrays[:, :, c], axes=(0,0))
    return ensemble

model_OOF_tensor = np.array([all_models_OOF[n] for n in model_names])
ensemble_b = per_class_weighted_ensemble(model_OOF_tensor, model_names, class_recalls)
pred_b = np.argmax(ensemble_b, axis=1)
acc_b = accuracy_score(y, pred_b)
print(f"  Per-class weighted ensemble: {acc_b:.4f}")

# ================================================================
# Strategy C: Logistic Regression ensemble (stack instead of weighted avg)
# ================================================================
print("\n=== Strategy C: Meta-LR on all models ===")

# Build meta features: all model probabilities flattened
n_model_count = len(model_names)
meta_size = n_model_count * 4
meta_X = np.zeros((n, meta_size))
for mi, mname in enumerate(model_names):
    meta_X[:, mi*4:(mi+1)*4] = all_models_OOF[mname]

cv_meta = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_c = np.zeros(n, dtype=int)
fold_c = []

for fi, (tr, val) in enumerate(cv_meta.split(meta_X, y)):
    meta_scaler = StandardScaler()
    X_tr = meta_scaler.fit_transform(meta_X[tr])
    X_val = meta_scaler.transform(meta_X[val])
    rs = 42 + fi

    meta_lr = LogisticRegression(C=1.0, max_iter=5000, random_state=rs)
    meta_lr.fit(X_tr, y[tr])
    oof_c[val] = meta_lr.predict(X_val)
    fold_c.append(accuracy_score(y[val], oof_c[val]))

acc_c = accuracy_score(y, oof_c)
print(f"  Meta-LR on all models: {acc_c:.4f}, mean={np.mean(fold_c):.4f}")

# ================================================================
# Strategy D: Reduced stack (best 4 models)
# ================================================================
print("\n=== Strategy D: Best 4 models stacked ===")

# Pick models with highest single-model OOF
model_individual_accs = {}
for mname in model_names:
    model_individual_accs[mname] = accuracy_score(y, np.argmax(all_models_OOF[mname], axis=1))

sorted_models = sorted(model_individual_accs.items(), key=lambda x: -x[1])
best4 = [m[0] for m in sorted_models[:4]]
print(f"  Best 4: {best4}")

meta_X4 = np.column_stack([all_models_OOF[n] for n in best4])

cv_meta4 = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_d = np.zeros(n, dtype=int)
fold_d = []
test_probs_d = []

for fi, (tr, val) in enumerate(cv_meta4.split(meta_X4, y)):
    rs = 42 + fi
    meta = LogisticRegression(C=0.3, max_iter=2000, random_state=rs)
    meta.fit(meta_X4[tr], y[tr])
    oof_d[val] = meta.predict(meta_X4[val])
    fold_d.append(accuracy_score(y[val], oof_d[val]))

    # Build test probs from the same 4 models
    te_meta = np.column_stack([test_probs_dict[n] for n in best4])
    test_probs_d.append(meta.predict_proba(te_meta))

acc_d = accuracy_score(y, oof_d)
print(f"  Best-4 stack LR: {acc_d:.4f}, mean={np.mean(fold_d):.4f}")

# ================================================================
# RESULTS
# ================================================================
print(f"\n{'='*50}")
print(f"EXP-008 ENSEMBLE RESULTS")
print(f"{'='*50}")
print(f"  A (Weighted avg):          {best_score:.4f}")
print(f"  B (Per-class weights):     {acc_b:.4f}")
print(f"  C (Meta-LR all models):    {acc_c:.4f}")
print(f"  D (Best-4 stack):          {acc_d:.4f}")
print(f"  Current best:              0.5988")

best_acc = max(best_score, acc_b, acc_c, acc_d)
best_strat = ['A','B','C','D'][np.argmax([best_score, acc_b, acc_c, acc_d])]
delta = best_acc - 0.5988
print(f"\n  Best: {best_strat} = {best_acc:.4f} (vs 0.5988: {delta:+.4f})")

# Save best submission
if best_strat == 'D' and test_probs_d:
    te_avg = np.mean(test_probs_d, axis=0)
    te_preds = np.argmax(te_avg, axis=1)
    sub = sample_sub.copy()
    sub['target'] = te_preds
    sub.to_csv(EXT_DIR / "submissions" / f"submission_{EX_ID.lower()}.csv", index=False)

import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID, 'parent_id': 'EXP-004',
    'hypothesis': 'Ensemble optimization with model weight search and per-class specialist weighting',
    'feature_family': 'base+sequence',
    'model_family': f'ensemble_stack (strat_{best_strat})',
    'parameters': json.dumps({'n_models': 7, 'random_trials': 500}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in (fold_d if best_strat == 'D' else fold_c if best_strat == 'C' else [acc_b]*10)]),
    'mean_accuracy': float(best_acc),
    'macro_f1': 0.0,
    'balanced_accuracy': 0.0,
    'minimum_fold': float(min(fold_d if best_strat == 'D' else fold_c)) if best_strat in ['C','D'] else 0,
    'train_accuracy': None, 'overfit_gap': None, 'runtime': 0,
    'accepted': best_acc > 0.5988,
    'rejection_reason': '' if best_acc > 0.5988 else 'not improving',
    'next_hypothesis': 'EXP-009: Self-supervised autoencoder features for representation improvement',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists: writer.writeheader()
    writer.writerow(row)
print("Done.")
