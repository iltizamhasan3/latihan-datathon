"""
EXP-001: Latent Score Reconstruction
Hypothesis: Target = bin(latent_score, thresholds) where latent_score is learned via
ordinal regression. Test with HGB regressor + optimized thresholds.
Followed by kelas group priors for refinement.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS

import warnings; warnings.filterwarnings('ignore')

SEED = 42
EX_ID = "EXP-001"
EXT_DIR = Path(__file__).parents[1]

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
y = train['target'].values
n = len(y)

# Build features
X = get_all_features(train).fillna(0).replace([np.inf, -np.inf], 0)
Xt = get_all_features(test).fillna(0).replace([np.inf, -np.inf], 0)

# Add sequence features
from scipy import stats as sp_stats
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

X_vals = X.fillna(0).replace([np.inf, -np.inf], 0).values
Xt_vals = Xt.fillna(0).replace([np.inf, -np.inf], 0).values

print(f"Features: {X_vals.shape[1]}")

# ================================================================
# STRATEGY A: Latent score via ordinal regression + thresholds
# ================================================================
print("\n=== Strategy A: Latent Score via HGB Regression + Thresholds ===")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_latent = np.zeros(n)
fold_scores_a = []

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])

    reg = HistGradientBoostingRegressor(max_iter=500, max_depth=3, learning_rate=0.05,
                                        random_state=42+fi)
    reg.fit(X_tr, y[tr])
    oof_latent[val] = reg.predict(X_val)

    # Optimize thresholds on this fold's train
    tr_latent = reg.predict(X_tr)
    best = 0
    best_t = None
    for t1 in np.percentile(tr_latent, [20, 25, 30, 33, 35]):
        for t2 in np.percentile(tr_latent, [45, 50, 55, 60, 66]):
            for t3 in np.percentile(tr_latent, [70, 75, 80, 85, 90]):
                if t1 < t2 < t3:
                    p = np.digitize(tr_latent, [t1, t2, t3])
                    a = (p == y[tr]).mean()
                    if a > best:
                        best = a
                        best_t = (t1, t2, t3)

    # Apply to validation
    val_pred = np.digitize(oof_latent[val], best_t)
    fa = accuracy_score(y[val], val_pred)
    fold_scores_a.append(fa)
    print(f"  Fold {fi+1}: acc={fa:.4f}, thresholds={np.round(best_t, 4)}")

print(f"Strategy A OOF: mean={np.mean(fold_scores_a):.4f}")
best_thresh_a = np.percentile(oof_latent, [25, 50, 75])
final_pred_a = np.digitize(oof_latent, best_thresh_a)
acc_a_full = accuracy_score(y, final_pred_a)
print(f"Strategy A full OOF: {acc_a_full:.4f}")

# ================================================================
# STRATEGY B: Weighted score from Lasso coefficients
# ================================================================
print("\n=== Strategy B: Lasso-weighted composite score ===")

from sklearn.linear_model import Lasso, LogisticRegressionCV

cv_b = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_b = np.zeros(n, dtype=int)
fold_scores_b = []

for fi, (tr, val) in enumerate(cv_b.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    X_te = scaler.transform(Xt_vals)

    # Multi-output regression: predict target as continuous
    lasso = Lasso(alpha=0.001, max_iter=5000, random_state=42+fi)
    lasso.fit(X_tr, y[tr])

    # Use regression output as latent score, find thresholds
    tr_score = lasso.predict(X_tr)
    val_score = lasso.predict(X_val)

    # Dynamic thresholds from training
    best = 0
    best_t = None
    for t1 in np.percentile(tr_score, [20, 25, 30, 33, 35]):
        for t2 in np.percentile(tr_score, [45, 50, 55, 60, 66]):
            for t3 in np.percentile(tr_score, [70, 75, 80, 85, 90]):
                if t1 < t2 < t3:
                    p = np.digitize(tr_score, [t1, t2, t3])
                    a = (p == y[tr]).mean()
                    if a > best:
                        best = a
                        best_t = (t1, t2, t3)

    if best_t:
        oof_b[val] = np.digitize(val_score, best_t)
        fold_scores_b.append(accuracy_score(y[val], oof_b[val]))
        print(f"  Fold {fi+1}: acc={fold_scores_b[-1]:.4f}")
    else:
        fold_scores_b.append(0.25)

mean_b = np.mean(fold_scores_b) if fold_scores_b else 0
print(f"Strategy B OOF: mean={mean_b:.4f}")

# ================================================================
# STRATEGY C: ElasicNet + Ridge + threshold blend
# ================================================================
print("\n=== Strategy C: Ensemble latent score ===")

from sklearn.linear_model import ElasticNet, Ridge

cv_c = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
fold_scores_c = []
oof_c = np.zeros(n, dtype=int)

for fi, (tr, val) in enumerate(cv_c.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])

    scores = []
    for model in [
        Ridge(alpha=1.0, random_state=42+fi),
        Ridge(alpha=0.5, random_state=42+fi),
        ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=42+fi),
    ]:
        model.fit(X_tr, y[tr])
        scores.append(model.predict(X_val))

    fitted_models = []
    for m_cls in [
        Ridge(alpha=1.0, random_state=42+fi),
        Ridge(alpha=0.5, random_state=42+fi),
        ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=42+fi),
    ]:
        m_cls.fit(X_tr, y[tr])
        fitted_models.append(m_cls)

    val_score = np.mean([m.predict(X_val) for m in fitted_models], axis=0)
    tr_score_mean = np.mean([m.predict(X_tr) for m in fitted_models], axis=0)

    best = 0; best_t = None
    for t1 in np.percentile(tr_score_mean, [20, 25, 30, 33, 35]):
        for t2 in np.percentile(tr_score_mean, [45, 50, 55, 60, 66]):
            for t3 in np.percentile(tr_score_mean, [70, 75, 80, 85, 90]):
                if t1 < t2 < t3:
                    p = np.digitize(tr_score_mean, [t1, t2, t3])
                    a = (p == y[tr]).mean()
                    if a > best:
                        best = a
                        best_t = (t1, t2, t3)

    if best_t:
        oof_c[val] = np.digitize(val_score, best_t)
        fold_scores_c.append(accuracy_score(y[val], oof_c[val]))
    else:
        fold_scores_c.append(0.25)

mean_c = np.mean(fold_scores_c) if fold_scores_c else 0
print(f"Strategy C OOF: mean={mean_c:.4f}")

# ================================================================
# RESULTS SUMMARY
# ================================================================
results = {
    'experiment_id': EX_ID,
    'hypothesis': 'Target reconstructed from latent regression score + optimal thresholds',
    'strategies': {
        'A_latent_hgb': {'mean': float(np.mean(fold_scores_a)), 'folds': [float(f) for f in fold_scores_a]},
        'B_lasso_linear': {'mean': float(mean_b), 'folds': [float(f) for f in fold_scores_b]},
        'C_ensemble_linear': {'mean': float(mean_c), 'folds': [float(f) for f in fold_scores_c]},
    },
}
best_strat = max(results['strategies'], key=lambda k: results['strategies'][k]['mean'])
results['best_strategy'] = best_strat
results['best_accuracy'] = results['strategies'][best_strat]['mean']

print(f"\n{'='*50}")
print(f"BEST STRATEGY: {best_strat} = {results['best_accuracy']:.4f}")
print(f"{'='*50}")

# Save as experiment log entry
import csv, os
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()

row = {
    'experiment_id': EX_ID,
    'parent_id': 'None',
    'hypothesis': 'Target = bin(latent_score, thresholds) via ordinal regression',
    'feature_family': 'base_engineered+sequence',
    'model_family': 'HGB_Regressor+Lasso+Ridge',
    'parameters': json.dumps({'max_iter':500, 'max_depth':3, 'threshold_search':'grid_20_90_percentile'}),
    'seed': str(SEED),
    'fold_scores': json.dumps(results['strategies'][best_strat]['folds']),
    'mean_accuracy': results['best_accuracy'],
    'macro_f1': 0.0,
    'balanced_accuracy': 0.0,
    'minimum_fold': min(results['strategies'][best_strat]['folds']),
    'train_accuracy': None,
    'overfit_gap': None,
    'runtime': 0,
    'accepted': False,
    'rejection_reason': '',
    'next_hypothesis': 'EXP-002: Kelas group priors + latent score refinement',
}

with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"\nLogged to {log_path}")
