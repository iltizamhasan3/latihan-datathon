"""
Phase 5: Pseudo-labeling + Threshold Optimization + Seed Averaging
"""
import numpy as np, pandas as pd, json, os, time, sys, warnings, copy
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from experiments.features import get_all_features

SEED = 42
DATA_DIR = Path("data")
EXP_DIR = Path("experiments")

best = {'accuracy': 0.5869, 'model': 'Stack_LR'}

def load_data():
    t = pd.read_csv(DATA_DIR / "train.csv"); te = pd.read_csv(DATA_DIR / "test.csv"); s = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return t, te, s

def cv_score(X, y, model, seed=SEED):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    accs = []
    for tr, val in cv.split(X, y):
        scaler = StandardScaler()
        m = copy.deepcopy(model)
        m.fit(scaler.fit_transform(X[tr]), y[tr])
        accs.append(accuracy_score(y[val], m.predict(scaler.transform(X[val]))))
    return np.mean(accs), np.std(accs)

def test_preds(X_full, y_full, X_test, model):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_full)
    X_te = scaler.transform(X_test)
    m = copy.deepcopy(model)
    m.fit(X_s, y_full)
    return m.predict(X_te)

# ================================================================
# SEED AVERAGING
# ================================================================
print("\n=== SEED AVERAGING ===")
train, test, sample_sub = load_data()
y = train['target'].values
X_full = get_all_features(train).fillna(0).replace([np.inf,-np.inf],0).values
X_te = get_all_features(test).fillna(0).replace([np.inf,-np.inf],0).values

# Models to try
models = [
    ("SVC_C50", lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s)),
    ("SVC_C10", lambda s: SVC(C=10, gamma='scale', probability=True, random_state=s)),
    ("LR_C1", lambda s: LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=s)),
]

print("\nSeed averaging results:")
for mname, mfn in models:
    scores = []
    for s in [42, 123, 2026, 777, 31415]:
        acc, std = cv_score(X_full, y, mfn(s), seed=s)
        scores.append(acc)
        print(f"  {mname:15s} seed={s:5d}: {acc:.4f}")
    print(f"  -> Mean across seeds: {np.mean(scores):.4f} (std={np.std(scores):.4f})")

# ================================================================
# PSEUDO-LABELING
# ================================================================
print("\n=== PSEUDO-LABELING ===")

# Use best combo model
mfn = lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s)

# 1. Get initial test predictions
te_pred = test_preds(X_full, y, X_te, mfn(SEED))

# 2. Find high-confidence predictions
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
te_probs = np.zeros((len(X_te), 4))
counts = np.zeros(len(X_te))

for tr, val in cv.split(X_full, y):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_full[tr])
    m = mfn(SEED)
    m.fit(X_tr, y[tr])
    te_probs += m.predict_proba(scaler.transform(X_te))
    counts += 1
te_probs /= counts
te_conf = te_probs.max(axis=1)
te_pred_final = te_probs.argmax(axis=1)

# High confidence threshold
for thresh in [0.90, 0.95, 0.97]:
    confident = te_conf >= thresh
    print(f"\n  Pseudo-label threshold={thresh:.2f}: {confident.sum()} samples confident")

    if confident.sum() < 10:
        print(f"  Too few confident, skipping")
        continue

    # Add pseudo-labels
    X_pseudo = np.vstack([X_full, X_te[confident]])
    y_pseudo = np.concatenate([y, te_pred_final[confident]])

    # Train with sample weights
    w = np.ones(len(y_pseudo))
    w[len(y):] = 0.3  # lower weight for pseudo-labels

    # Evaluate
    cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    accs = []
    for tr, val in cv2.split(X_pseudo, y_pseudo):
        scaler = StandardScaler()
        m = mfn(42)
        m.fit(scaler.fit_transform(X_pseudo[tr]), y_pseudo[tr])
        preds = m.predict(scaler.transform(X_pseudo[val]))
        # Only evaluate on original data
        orig_mask = val < len(y)
        if orig_mask.sum() > 0:
            accs.append(accuracy_score(y_pseudo[val[orig_mask]], preds[orig_mask]))

    if accs:
        pl_acc = np.mean(accs)
        print(f"  Pseudo-label OOF: {pl_acc:.4f}")
        baseline, _ = cv_score(X_full, y, mfn(42))
        print(f"  Baseline: {baseline:.4f}, Gain: {pl_acc - baseline:.4f}")

# ================================================================
# THRESHOLD OPTIMIZATION
# ================================================================
print("\n=== THRESHOLD OPTIMIZATION ===")
from sklearn.metrics import accuracy_score

# Get OOF predictions for threshold optimization
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
oof_probs = np.zeros((len(y), 4))
for tr, val in cv.split(X_full, y):
    scaler = StandardScaler()
    m = mfn(42)
    m.fit(scaler.fit_transform(X_full[tr]), y[tr])
    oof_probs[val] = m.predict_proba(scaler.transform(X_full[val]))

# Try shifting thresholds
default_acc = accuracy_score(y, oof_probs.argmax(axis=1))
print(f"  Default argmax accuracy: {default_acc:.4f}")

best_shift = 0
best_thresh_acc = 0
for shift in np.arange(-0.3, 0.31, 0.05):
    shifted = oof_probs.copy()
    shifted[:, 1] += shift; shifted[:, 2] -= shift * 0.5
    shifted = np.clip(shifted, 0.01, 0.99)
    shifted /= shifted.sum(axis=1, keepdims=True)
    preds = shifted.argmax(axis=1)
    acc = accuracy_score(y, preds)
    if acc > best_thresh_acc:
        best_thresh_acc = acc
        best_shift = shift

print(f"  Best threshold shift: {best_shift:.2f} -> {best_thresh_acc:.4f}")
print(f"  Improvement: +{best_thresh_acc - default_acc:.4f}")

# ================================================================
# BEST OVERALL PREDICTION
# ================================================================
print(f"\n{'='*60}")
print("FINAL ENSEMBLE PREDICTION")
print(f"{'='*60}")

# Ensemble: SVC C50 + SVC C10 + LR across multiple seeds
n_models = 0
final_probs = np.zeros((len(X_te), 4))

for mname, mfn in models:
    for s in [42, 123, 2026]:
        scaler = StandardScaler()
        m = mfn(s)
        m.fit(scaler.fit_transform(X_full), y)
        final_probs += m.predict_proba(scaler.transform(X_te))
        n_models += 1
        print(f"  Added {mname}(seed={s})")

final_probs /= n_models
final_preds = final_probs.argmax(axis=1)

# Apply best threshold shift if useful
if best_shift != 0 and best_thresh_acc > default_acc + 0.001:
    final_probs[:, 1] += best_shift
    final_probs[:, 2] -= best_shift * 0.5
    final_probs = np.clip(final_probs, 0.01, 0.99)
    final_probs /= final_probs.sum(axis=1, keepdims=True)
    final_preds = final_probs.argmax(axis=1)

sub = sample_sub.copy()
sub['target'] = final_preds.astype(int)
sub.to_csv(EXP_DIR / "submission_final_ensemble.csv", index=False)
print(f"\nSaved: submission_final_ensemble.csv")
print(f"Target distribution: {np.bincount(final_preds.astype(int))}")
print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
