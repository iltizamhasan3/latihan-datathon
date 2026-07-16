"""
EXP-012: Synthesis — combine everything that gave positive signal.
1. Full engineered + sequence features
2. Per-fold OOF from SVC(C=10) + CatBoost(depth=4)
3. Meta: LogisticRegression(C=0.3) on stacked OOF probs
4. Multi-seed verification if above best
Uses exact setup from EXP-004 best config.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats as sp_stats

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
import catboost as cb

import warnings; warnings.filterwarnings('ignore')
SEED = 42; EX_ID = "EXP-012"
EXT_DIR = Path(__file__).parents[1]

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
y = train['target'].values; n = len(y)

# Build features
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
print(f"Features: {X_vals.shape[1]}")

# ================================================================
# Best config from EXP-004: SVC(C=10), CB(depth=4), meta LR(C=0.3)
# Primary CV: RepeatedStratifiedKFold(5, 2)
# ================================================================
print("\n=== Primary: Best Config [SVC(C=10)+CB(d=4), meta LR(C=0.3)] ===")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof = np.zeros(n, dtype=int); folds = []; test_probs = []

t0 = time.time()
for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr]); X_val = scaler.transform(X_vals[val]); X_te = scaler.transform(Xt_vals)
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
    print(f"  Fold {fi+1}: acc={folds[-1]:.4f} ({time.time()-t0:.0f}s)")

acc = accuracy_score(y, oof)
f1 = f1_score(y, oof, average='macro')
runtime = time.time() - t0
print(f"\n  OOF: {acc:.4f}, f1={f1:.4f}, mean={np.mean(folds):.4f}")

# ================================================================
# Multi-seed verification
# ================================================================
print("\n=== Multi-Seed Verification ===")
seeds = [42, 123, 2026, 3407, 7777]
seed_scores = []
for s in seeds:
    cv_s = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=s)
    oof_s = np.zeros(n, dtype=int)
    for fi, (tr, val) in enumerate(cv_s.split(X_vals, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_vals[tr]); X_val = scaler.transform(X_vals[val])
        rs = 42 + fi
        svc = SVC(C=10, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
        cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0)
        cb_m.fit(X_tr, y[tr])
        meta_val = np.column_stack([svc.predict_proba(X_val), cb_m.predict_proba(X_val)])
        meta = LogisticRegression(C=0.3, max_iter=2000, random_state=rs)
        meta.fit(meta_val, y[val])
        oof_s[val] = meta.predict(meta_val)
    sa = accuracy_score(y, oof_s)
    seed_scores.append(sa)
    print(f"  Seed {s}: {sa:.4f}")

# ================================================================
# Nested CV
# ================================================================
print("\n=== Nested CV ===")
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
outer_preds = np.zeros(n, dtype=int)
outer_scores = []

for oi, (otr, oval) in enumerate(outer_cv.split(X_vals, y)):
    inner_oof = {}
    for mname, mfn in [('SVC', lambda rs: SVC(C=10, gamma='auto', probability=True, random_state=rs)),
                        ('CB', lambda rs: cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0))]:
        oof_p = np.zeros((len(otr), 4))
        for ir, iv in inner_cv.split(X_vals[otr], y[otr]):
            sc = StandardScaler()
            X_ir = sc.fit_transform(X_vals[otr][ir]); X_iv = sc.transform(X_vals[otr][iv])
            m = mfn(42).fit(X_ir, y[otr][ir])
            oof_p[iv] = m.predict_proba(X_iv)
        inner_oof[mname] = oof_p

    X_meta = np.column_stack([inner_oof['SVC'], inner_oof['CB']])
    meta_o = LogisticRegression(C=0.3, max_iter=2000, random_state=42)
    meta_o.fit(X_meta, y[otr])

    sc_o = StandardScaler()
    X_otr_f = sc_o.fit_transform(X_vals[otr]); X_oval = sc_o.transform(X_vals[oval])
    ob = np.column_stack([
        SVC(C=10, gamma='auto', probability=True, random_state=42).fit(X_otr_f, y[otr]).predict_proba(X_oval),
        cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=42, verbose=0).fit(X_otr_f, y[otr]).predict_proba(X_oval)
    ])
    outer_preds[oval] = meta_o.predict(ob)
    outer_scores.append(accuracy_score(y[oval], outer_preds[oval]))
    print(f"  Outer fold {oi+1}: {outer_scores[-1]:.4f}")

nested_acc = accuracy_score(y, outer_preds)
print(f"\n  Nested CV: {nested_acc:.4f}")

# ================================================================
# Final Results
# ================================================================
print(f"\n{'='*50}")
print(f"EXP-012 FINAL RESULTS")
print(f"{'='*50}")
print(f"  OOF accuracy:         {acc:.4f}")
print(f"  Macro-F1:             {f1:.4f}")
print(f"  Multi-seed mean:      {np.mean(seed_scores):.4f}")
print(f"  Multi-seed std:       {np.std(seed_scores):.4f}")
print(f"  Multi-seed min:       {min(seed_scores):.4f}")
print(f"  Nested CV:            {nested_acc:.4f}")
print(f"  Current best (EXP-4): 0.5988")
delta = acc - 0.5988
print(f"  Delta:                {delta:+.4f}")

# Save
test_avg = np.mean(test_probs, axis=0)
test_preds = np.argmax(test_avg, axis=1)
sub = sample_sub.copy(); sub['target'] = test_preds
sub.to_csv(EXT_DIR / "submissions" / f"submission_{EX_ID.lower()}.csv", index=False)

import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID, 'parent_id': 'EXP-004',
    'hypothesis': 'Best config verified with multi-seed + nested CV: SVC(C=10)+CB(d=4)+LR(C=0.3)',
    'feature_family': 'base_engineered+sequence',
    'model_family': 'Stack_SVC(C=10)+CB(d=4)_LR(C=0.3)',
    'parameters': json.dumps({'svc_C': 10, 'cb_depth': 4, 'cb_n': 400, 'meta_C': 0.3, 'n_repeats': 2}),
    'seed': '42,123,2026,3407,7777',
    'fold_scores': json.dumps([float(f) for f in folds]),
    'mean_accuracy': float(acc),
    'macro_f1': float(f1),
    'balanced_accuracy': 0.0,
    'minimum_fold': float(min(folds)),
    'train_accuracy': None, 'overfit_gap': None, 'runtime': float(runtime),
    'accepted': acc > 0.5975,
    'rejection_reason': '',
    'next_hypothesis': 'Extreme search complete — initiating final report synthesis',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists: writer.writeheader()
    writer.writerow(row)

# Save full results
result = {
    'best_oof': float(acc), 'f1': float(f1), 'fold_scores': [float(f) for f in folds],
    'multi_seed_mean': float(np.mean(seed_scores)), 'multi_seed_std': float(np.std(seed_scores)),
    'multi_seed_min': float(min(seed_scores)), 'multi_seed_scores': [float(s) for s in seed_scores],
    'nested_cv': float(nested_acc), 'runtime': float(runtime),
}
with open(EXT_DIR / "oof" / "best_verified_results.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nDone. Results saved.")
