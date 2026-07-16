"""
EXP-004: Aggressive stacking optimization around best known pipeline.
Hypothesis: Systematic sweep of SVC/CatBoost/HGB/Ordinal parameters +
kelas meta-features + advanced sequence can push past 0.60.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats as sp_stats

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
import catboost as cb

import warnings; warnings.filterwarnings('ignore')

SEED = 42
EX_ID = "EXP-004"
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
tr_kelas = train['kelas'].fillna(-1).astype(int).values
te_kelas = test['kelas'].fillna(-1).astype(int).values

print(f"Features: {X_vals.shape[1]}")

def kelas_priors(tr_k, te_k, y_tr):
    """Compute per-kelas target priors."""
    uniq = np.unique(tr_k)
    priors = {}
    for k in uniq:
        m = tr_k == k
        priors[k] = np.array([(y_tr[m]==c).mean() for c in range(4)]) if m.sum() > 0 else np.array([0.25]*4)
    return priors

# ================================================================
# 4-model stacking sweep: SVC, CatBoost, HGB, ExtraTrees
# ================================================================
print("\n=== 4-Model Stacking Sweep ===")

best_overall = 0
best_config = None

# Sweep key parameters
svc_C_vals = [10, 50, 100]
catboost_depth = [4, 6]
meta_C_vals = [0.05, 0.1, 0.3]

for svc_C in svc_C_vals:
    for cb_depth in catboost_depth:
        for meta_C in meta_C_vals:
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
            oof = np.zeros(n, dtype=int)
            folds = []
            test_probs = []

            for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_vals[tr])
                X_val = scaler.transform(X_vals[val])
                X_te = scaler.transform(Xt_vals)
                rs = 42 + fi

                # Kelas priors
                kp = kelas_priors(tr_kelas[tr], te_kelas, y[tr])
                val_prior = np.array([kp.get(k, [0.25]*4) for k in tr_kelas[val]])
                te_prior = np.array([kp.get(k, [0.25]*4) for k in te_kelas])

                # Base models
                svc = SVC(C=svc_C, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
                cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=cb_depth, learning_rate=0.05, random_seed=rs, verbose=0)
                cb_m.fit(X_tr, y[tr])
                hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.05, random_state=rs)
                hgb.fit(X_tr, y[tr])
                et = ExtraTreesClassifier(n_estimators=400, max_depth=6, random_state=rs)
                et.fit(X_tr, y[tr])

                # OOF probs
                meta_val = np.column_stack([
                    svc.predict_proba(X_val), cb_m.predict_proba(X_val),
                    hgb.predict_proba(X_val), et.predict_proba(X_val),
                    val_prior
                ])
                meta_te = np.column_stack([
                    svc.predict_proba(X_te), cb_m.predict_proba(X_te),
                    hgb.predict_proba(X_te), et.predict_proba(X_te),
                    te_prior
                ])

                meta = LogisticRegression(C=meta_C, max_iter=2000, random_state=rs)
                meta.fit(meta_val, y[val])
                oof[val] = meta.predict(meta_val)
                folds.append(accuracy_score(y[val], oof[val]))
                test_probs.append(meta.predict_proba(meta_te))

            acc = accuracy_score(y, oof)
            if acc > best_overall:
                best_overall = acc
                best_config = f"SVC_{svc_C}_CB{cb_depth}_meta{meta_C}"
                best_folds = folds
                best_test = test_probs
                best_oof = oof

            print(f"  SVC_C={svc_C}, CB_d={cb_depth}, meta_C={meta_C}: {acc:.4f} (mean={np.mean(folds):.4f})")

print(f"\nBest: {best_config} = {best_overall:.4f}")
delta = best_overall - 0.5975
print(f"  vs 0.5975: {delta:+.4f}")

# ================================================================
# Save results
# ================================================================
if best_test:
    test_avg = np.mean(best_test, axis=0)
    test_preds = np.argmax(test_avg, axis=1)
    sub = sample_sub.copy()
    sub['target'] = test_preds
    sub.to_csv(EXT_DIR / "submissions" / f"submission_{EX_ID.lower()}.csv", index=False)

import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID,
    'parent_id': 'EXP-002',
    'hypothesis': '4-model stacking (SVC+CatBoost+HGB+ET) + kelas priors with parameter sweep',
    'feature_family': 'base_engineered+sequence+kelas_priors',
    'model_family': f'Stack_LR({best_config})',
    'parameters': json.dumps({'sweep': 'svc_C_10-100, cb_depth_4-6, meta_C_0.05-0.3', 'best': best_config}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in best_folds]),
    'mean_accuracy': float(best_overall),
    'macro_f1': float(f1_score(y, best_oof, average='macro')),
    'balanced_accuracy': 0.0,
    'minimum_fold': float(min(best_folds)),
    'train_accuracy': None,
    'overfit_gap': None,
    'runtime': 0,
    'accepted': best_overall > 0.5975,
    'rejection_reason': '' if best_overall > 0.5975 else 'at or below current best',
    'next_hypothesis': 'EXP-005: Ordinal stacking with CORAL/cumulative link models',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists:
        writer.writeheader()
    writer.writerow(row)
print(f"\nDone.")
