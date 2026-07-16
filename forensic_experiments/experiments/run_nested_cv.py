"""
Nested CV validation for the best pipeline found so far.
Uses the sequence-stacking pipeline that gave 0.5975.
"""
import numpy as np, pandas as pd, json, sys, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.append('/home/user/projects/latihan-datathon')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
from forensic_experiments.core import FOR_DIR

import catboost as cb

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
y = train['target'].values

# Build base features
X = get_all_features(train).fillna(0).replace([np.inf, -np.inf], 0)
Xt = get_all_features(test).fillna(0).replace([np.inf, -np.inf], 0)

# Build sequence features (same as Phase 5)
from scipy import stats as sp_stats
for phase_prefix, cols, n_cols in [('week', WEEK_COLS, 12), ('act', ACTIVITY_COLS, 16)]:
    train_seq = train[cols].fillna(0).values
    test_seq = test[cols].fillna(0).values
    pf = phase_prefix
    x_axis = np.arange(n_cols)

    for seq_data, prefix in [(train_seq, 'train'), (test_seq, 'test')]:
        slopes = np.array([sp_stats.linregress(x_axis, seq_data[i])[0] for i in range(len(seq_data))])
        X if prefix == 'train' else Xt
        if prefix == 'train':
            X[f'seq_{pf}_slope_robust'] = slopes
        else:
            Xt[f'seq_{pf}_slope_robust'] = slopes

        # Acceleration
        accel = np.array([np.polyfit(x_axis, seq_data[i], 2)[0] * 2 for i in range(len(seq_data))])
        if prefix == 'train':
            X[f'seq_{pf}_acceleration'] = accel
            X[f'seq_{pf}_curvature'] = np.abs(accel) / (1 + slopes**2)**1.5
        else:
            Xt[f'seq_{pf}_acceleration'] = accel
            Xt[f'seq_{pf}_curvature'] = np.abs(accel) / (1 + slopes**2)**1.5

        # Autocorrelation
        autocorr = np.array([np.corrcoef(seq_data[i, :-1], seq_data[i, 1:])[0, 1]
                            if np.std(seq_data[i, :-1]) > 0 and np.std(seq_data[i, 1:]) > 0 else 0
                            for i in range(len(seq_data))])
        if prefix == 'train':
            X[f'seq_{pf}_autocorr_lag1'] = autocorr
        else:
            Xt[f'seq_{pf}_autocorr_lag1'] = autocorr

        # Spectral entropy
        fft_vals = np.abs(np.fft.fft(seq_data, axis=1))
        fft_power = fft_vals[:, :n_cols//2] ** 2
        fft_norm = fft_power / (fft_power.sum(axis=1, keepdims=True) + 1e-10)
        spectral_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10), axis=1)
        if prefix == 'train':
            X[f'seq_{pf}_spectral_entropy'] = spectral_entropy
        else:
            Xt[f'seq_{pf}_spectral_entropy'] = spectral_entropy

        # Change point count
        diffs = np.diff(seq_data, axis=1)
        sign_changes = np.sum(np.diff(np.sign(diffs), axis=1) != 0, axis=1)
        if prefix == 'train':
            X[f'seq_{pf}_change_points'] = sign_changes
        else:
            Xt[f'seq_{pf}_change_points'] = sign_changes

X_vals = X.fillna(0).replace([np.inf, -np.inf], 0).values
Xt_vals = Xt.fillna(0).replace([np.inf, -np.inf], 0).values

print(f"Features with sequence: {X_vals.shape}")

# ================================================================
# NESTED CV
# ================================================================
print("\n" + "="*60)
print("NESTED CROSS-VALIDATION (Best pipeline)")
print("="*60)

def get_ordinal_probs(X_train, y_train, X_val, X_test):
    probs_val = np.zeros((X_val.shape[0], 4))
    probs_test = np.zeros((X_test.shape[0], 4))
    for k in range(1, 4):
        y_bin = (y_train >= k).astype(int)
        m = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
        m.fit(X_train, y_bin)
        probs_val[:, k] = m.predict_proba(X_val)[:, 1]
        probs_test[:, k] += m.predict_proba(X_test)[:, 1]
    probs_val[:, 0] = 1.0 - probs_val[:, 1]
    return probs_val / probs_val.sum(axis=1, keepdims=True), probs_test / probs_test.sum(axis=1, keepdims=True)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
n = len(y)
outer_preds = np.zeros(n, dtype=int)
outer_fold_scores = []
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

start = time.time()

for oi, (otr, oval) in enumerate(outer_cv.split(X_vals, y)):
    print(f"\n  Outer fold {oi+1}/5 ({len(otr)} train, {len(oval)} val)")

    # Inner: generate OOF probs for SVC + CatBoost + Ordinal on otr
    models_inner = {
        'SVC': lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
        'CatBoost': lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0),
    }

    inner_oof = {}
    for mname, mfn in models_inner.items():
        oof_p = np.zeros((len(otr), 4))
        for tr, val in inner_cv.split(X_vals[otr], y[otr]):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_vals[otr][tr])
            X_val = sc.transform(X_vals[otr][val])
            m = mfn(42)
            m.fit(X_tr, y[otr][tr])
            oof_p[val] = m.predict_proba(X_val)
        inner_oof[mname] = oof_p

    # Ordinal inner OOF
    scaler_o = StandardScaler()
    X_otr_scaled = scaler_o.fit_transform(X_vals[otr])
    oof_ord = np.zeros((len(otr), 4))
    for tr, val in inner_cv.split(X_vals[otr], y[otr]):
        sc = StandardScaler()
        X_inner_tr = sc.fit_transform(X_vals[otr][tr])
        X_inner_val = sc.transform(X_vals[otr][val])
        oof_ord[val], _ = get_ordinal_probs(X_inner_tr, y[otr][tr], X_inner_val, X_inner_val[:1])  # placeholder, not used
    inner_oof['Ordinal'] = oof_ord

    # Meta model
    X_meta_train = np.column_stack([inner_oof[n] for n in ['SVC', 'CatBoost']])
    meta = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    meta.fit(X_meta_train, y[otr])

    # Predict outer val
    sc_outer = StandardScaler()
    X_otr_final = sc_outer.fit_transform(X_vals[otr])
    X_oval_final = sc_outer.transform(X_vals[oval])

    outer_base = np.column_stack([
        models_inner['SVC'](42).fit(X_otr_final, y[otr]).predict_proba(X_oval_final),
        models_inner['CatBoost'](42).fit(X_otr_final, y[otr]).predict_proba(X_oval_final),
    ])

    outer_preds[oval] = meta.predict(outer_base)
    outer_acc = accuracy_score(y[oval], outer_preds[oval])
    outer_fold_scores.append(outer_acc)
    print(f"    -> Outer fold acc: {outer_acc:.4f}")

nested_acc = accuracy_score(y, outer_preds)
nested_f1 = f1_score(y, outer_preds, average='macro')
elapsed = time.time() - start

print(f"\n{'='*60}")
print(f"NESTED CV RESULTS")
print(f"{'='*60}")
print(f"  Nested CV accuracy: {nested_acc:.4f}")
print(f"  Nested CV macro-F1: {nested_f1:.4f}")
print(f"  Outer fold scores: {[f'{s:.4f}' for s in outer_fold_scores]}")
print(f"  Mean outer fold: {np.mean(outer_fold_scores):.4f}")
print(f"  Std outer fold: {np.std(outer_fold_scores):.4f}")
print(f"  Standard OOF best: 0.5975")
print(f"  Gap: {0.5975 - nested_acc:+.4f}")
print(f"  Runtime: {elapsed:.0f}s")

# Save
nested_results = {
    'timestamp': f'{pd.Timestamp.now()}',
    'experiment': 'Nested CV Stack_LR_C0.1+Ordinal+seq',
    'nested_accuracy': float(nested_acc),
    'nested_f1': float(nested_f1),
    'outer_fold_scores': [float(s) for s in outer_fold_scores],
    'mean_outer': float(np.mean(outer_fold_scores)),
    'std_outer': float(np.std(outer_fold_scores)),
    'standard_oof_best': 0.5975,
    'gap_vs_standard': float(0.5975 - nested_acc),
    'runtime': elapsed
}

with open(FOR_DIR / "model_analysis" / "nested_cv_results.json", 'w') as f:
    json.dump(nested_results, f, indent=2, default=str)

# Also run standard OOF for the same pipeline to get clean comparison
print(f"\n--- Standard OOF for comparison ---")
cv_std = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
models_std = {
    'SVC': lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
    'CatBoost': lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0),
}
all_oof_s = {}
for mname, mfn in models_std.items():
    oof_p = np.zeros((n, 4))
    for tr, val in cv_std.split(X_vals, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_vals[tr])
        X_val = sc.transform(X_vals[val])
        m = mfn(42)
        m.fit(X_tr, y[tr])
        oof_p[val] = m.predict_proba(X_val)
    all_oof_s[mname] = oof_p

X_meta_s = np.column_stack([all_oof_s[n] for n in ['SVC', 'CatBoost']])
for C in [0.1, 0.5, 1.0]:
    meta_s = LogisticRegression(C=C, max_iter=2000, random_state=42)
    oof_s = np.zeros(n, dtype=int)
    for tr, val in cv_std.split(X_vals, y):
        m = LogisticRegression(C=C, max_iter=2000, random_state=42)
        m.fit(X_meta_s[tr], y[tr])
        oof_s[val] = m.predict(X_meta_s[val])
    acc = accuracy_score(y, oof_s)
    print(f"  Stack_LR(C={C}) standard OOF: {acc:.4f}")

print(f"\nDone.")
