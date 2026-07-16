"""
EXP-009: Self-supervised autoencoder features.
Train denoising autoencoder on all data (train+test, unsupervised),
extract bottleneck features, add to main model.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats as sp_stats

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
import catboost as cb

import warnings; warnings.filterwarnings('ignore')
SEED = 42; EX_ID = "EXP-009"
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

X_vals = X.values.astype(np.float64); Xt_vals = Xt.values.astype(np.float64)
print(f"Features: {X_vals.shape[1]}")

# ================================================================
# Build autoencoder features
# ================================================================
print("\n=== Autoencoder Feature Learning ===")

from sklearn.neural_network import MLPRegressor

# Standardize all data (unsupervised - safe for transductive)
all_data = np.vstack([X_vals, Xt_vals])
scaler_all = StandardScaler()
all_scaled = scaler_all.fit_transform(all_data)

# Train autoencoder on ALL data (train + test, unsupervised)
# Simple stacked denoising autoencoder
print("Training autoencoder on all data (unsupervised)...")

# Add noise for denoising
noise_factor = 0.1
all_noisy = all_scaled + np.random.RandomState(SEED).normal(0, noise_factor, all_scaled.shape)

# Layer 1: 126 -> 64, 64 -> 126
encoder1 = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', max_iter=500,
                         random_state=SEED, verbose=False)
encoder1.fit(all_noisy, all_scaled)
enc1_features = encoder1.predict(all_scaled)  # Actually reconstruction
# Better: get hidden layer activations
from sklearn.pipeline import Pipeline

# Train again, simpler approach: PCA reduction + nonlinear
# PCA baseline
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=SEED)
pca_feats = pca.fit_transform(all_scaled)

# KPCA for nonlinear
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=20, kernel='rbf', gamma=0.05, random_state=SEED)
kpca_feats = kpca.fit_transform(all_scaled)

# Train actual denoising AE using MLP
ae_encoder = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', max_iter=1000,
                           random_state=SEED, warm_start=True, early_stopping=True)
# First learn identity with bottleneck
X_ae = all_scaled.copy()
ae_encoder.fit(X_ae, X_ae)

# Extract bottleneck features by predicting with intermediate layer?
# Simple: use the MLP's hidden representation
# MLP doesn't expose hidden layers directly without hacks.
# Use simpler approach: train to reconstruct, use residuals as features

ae_residuals = X_ae - ae_encoder.predict(X_ae)

# Use PCA + KPCA + residuals as features
combined_feats = np.column_stack([pca_feats, kpca_feats, ae_residuals])

X_ae_feats = combined_feats[:n]
Xt_ae_feats = combined_feats[n:]

print(f"Autoencoder features: {X_ae_feats.shape[1]}")
print(f"  PCA components: {pca_feats.shape[1]}")
print(f"  KPCA components: 20")
print(f"  AE residuals: {ae_residuals.shape[1]}")

# ================================================================
# Stack with autoencoder features
# ================================================================
print("\n=== Stacking with Autoencoder Features ===")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
X_full = np.column_stack([X_vals, X_ae_feats])
Xt_full = np.column_stack([Xt_vals, Xt_ae_feats])

oof = np.zeros(n, dtype=int)
folds = []
test_probs = []

for fi, (tr, val) in enumerate(cv.split(X_full, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_full[tr])
    X_val = scaler.transform(X_full[val])
    X_te = scaler.transform(Xt_full)
    rs = 42 + fi

    svc = SVC(C=50, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
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
print(f"AE features stack: {acc:.4f}, mean={np.mean(folds):.4f}")

# ================================================================
# RESULTS
# ================================================================
delta = acc - 0.5988
print(f"\n{'='*50}")
print(f"EXP-009 AE FEATURES")
print(f"{'='*50}")
print(f"  OOF: {acc:.4f} (vs 0.5988: {delta:+.4f})")

import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID, 'parent_id': 'EXP-004',
    'hypothesis': 'Self-supervised autoencoder features (PCA+KPCA+AE residuals) improve representation',
    'feature_family': 'base+sequence+PCA+KPCA+AE_residuals',
    'model_family': 'Stack_SVC+CB_LR',
    'parameters': json.dumps({'pca_0.95': pca_feats.shape[1], 'kpca_20': True, 'ae_hidden': 32}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in folds]),
    'mean_accuracy': float(acc),
    'macro_f1': 0.0,
    'balanced_accuracy': 0.0,
    'minimum_fold': float(min(folds)),
    'train_accuracy': None, 'overfit_gap': None, 'runtime': 0,
    'accepted': acc > 0.5988,
    'rejection_reason': '' if acc > 0.5988 else 'not improving',
    'next_hypothesis': 'EXP-010: Deep feature interactions (top pairwise products selected by tree-based screening)',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists: writer.writeheader()
    writer.writerow(row)
print("Done.")
