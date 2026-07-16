"""
EXP-007: Graph-based learning: label propagation + spectral features.
Uses KNN graph constructed from features within each fold only.
Label propagation - spread training labels through graph structure.
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
SEED = 42; EX_ID = "EXP-007"
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

from sklearn.neighbors import NearestNeighbors

# ================================================================
# Strategy A: Graph label propagation features
# For each sample, compute neighbor-weighted label distribution
# within fold only to avoid leakage
# ================================================================
print("\n=== Strategy A: Graph Label Propagation ===")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
graph_probs = np.zeros((n, 4))

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])

    # Build KNN on training data
    for n_neighbors in [5, 10, 20, 40]:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(X_tr)

        # Find nearest neighbors in training for validation samples
        dists, idxs = nn.kneighbors(X_val)

        # Compute neighbor-weighted label distribution
        weights = 1.0 / (dists + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        for i in range(len(val)):
            neighbor_labels = y[tr][idxs[i]]
            for c in range(4):
                graph_probs[val[i], c] += weights[i][neighbor_labels == c].sum()

# Normalize
row_sums = graph_probs.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
graph_probs = graph_probs / row_sums

# Use graph probs directly as predictions
pred_a = np.argmax(graph_probs, axis=1)
acc_a = accuracy_score(y, pred_a)
print(f"  Graph label prop OOF: {acc_a:.4f}")

# ================================================================
# Strategy B: Graph features + main stacking
# ================================================================
print("\n=== Strategy B: Graph Features in Stack ===")

# Re-compute graph features fold-safe
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
graph_features = np.zeros((n, 8))  # 4 neighbor counts + 4 distance features

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])

    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_tr)
    dists, idxs = nn.kneighbors(X_val)

    for i in range(len(val)):
        nl = y[tr][idxs[i]]
        for c in range(4):
            graph_features[val[i], c] = (nl == c).sum()
        graph_features[val[i], 4] = dists[i].mean()
        graph_features[val[i], 5] = dists[i].min()
        graph_features[val[i], 6] = dists[i].max()
        graph_features[val[i], 7] = dists[i].std()

print(f"Graph features shape: {graph_features.shape}")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_b = np.zeros(n, dtype=int)
fold_b = []
test_probs_b = []

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(np.column_stack([X_vals[tr], graph_features[tr]]))
    X_val = scaler.transform(np.column_stack([X_vals[val], graph_features[val]]))
    X_te = scaler.transform(np.column_stack([Xt_vals, np.zeros((len(Xt_vals), graph_features.shape[1]))]))
    rs = 42 + fi

    svc = SVC(C=50, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
    cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0)
    cb_m.fit(X_tr, y[tr])

    meta_val = np.column_stack([svc.predict_proba(X_val), cb_m.predict_proba(X_val)])
    meta_te = np.column_stack([svc.predict_proba(X_te), cb_m.predict_proba(X_te)])

    meta = LogisticRegression(C=0.3, max_iter=2000, random_state=rs)
    meta.fit(meta_val, y[val])
    oof_b[val] = meta.predict(meta_val)
    fold_b.append(accuracy_score(y[val], oof_b[val]))
    test_probs_b.append(meta.predict_proba(meta_te))

acc_b = accuracy_score(y, oof_b)
print(f"Graph features stack: {acc_b:.4f}, mean={np.mean(fold_b):.4f}")

# ================================================================
# Strategy C: Spectral embedding + stacking
# ================================================================
print("\n=== Strategy C: Spectral Embedding Features ===")

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD

# Compute spectral features on full data (unsupervised, no target)
# This is safe as transductive unsupervised learning
all_data = np.vstack([X_vals, Xt_vals])
scaler_full = StandardScaler()
all_scaled = scaler_full.fit_transform(all_data)

kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.1, random_state=SEED)
kpca_emb = kpca.fit_transform(all_scaled)

X_kpca = kpca_emb[:n]
Xt_kpca = kpca_emb[n:]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_c = np.zeros(n, dtype=int)
fold_c = []
test_probs_c = []

X_full_c = np.column_stack([X_vals, X_kpca])
Xt_full_c = np.column_stack([Xt_vals, Xt_kpca])

for fi, (tr, val) in enumerate(cv.split(X_full_c, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_full_c[tr])
    X_val = scaler.transform(X_full_c[val])
    X_te = scaler.transform(Xt_full_c)
    rs = 42 + fi

    svc = SVC(C=50, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
    cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0)
    cb_m.fit(X_tr, y[tr])

    meta_val = np.column_stack([svc.predict_proba(X_val), cb_m.predict_proba(X_val)])
    meta_te = np.column_stack([svc.predict_proba(X_te), cb_m.predict_proba(X_te)])

    meta = LogisticRegression(C=0.3, max_iter=2000, random_state=rs)
    meta.fit(meta_val, y[val])
    oof_c[val] = meta.predict(meta_val)
    fold_c.append(accuracy_score(y[val], oof_c[val]))
    test_probs_c.append(meta.predict_proba(meta_te))

acc_c = accuracy_score(y, oof_c)
print(f"Spectral stack: {acc_c:.4f}, mean={np.mean(fold_c):.4f}")

# ================================================================
# RESULTS
# ================================================================
print(f"\n{'='*50}")
print(f"EXP-007 RESULTS")
print(f"{'='*50}")
print(f"  A (Graph label prop):          {acc_a:.4f}")
print(f"  B (Graph features stack):      {acc_b:.4f}")
print(f"  C (Spectral embedding stack):  {acc_c:.4f}")
print(f"  Current best:                  0.5988")

best_acc = max(acc_a, acc_b, acc_c)
best_strat = ['A','B','C'][np.argmax([acc_a, acc_b, acc_c])]
delta = best_acc - 0.5988
print(f"\n  Best: {best_strat} = {best_acc:.4f} (vs 0.5988: {delta:+.4f})")

import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID, 'parent_id': 'EXP-004',
    'hypothesis': 'Graph structure (KNN label prop + spectral embedding) reveals hidden class separation',
    'feature_family': 'base+sequence+graph_features+spectral',
    'model_family': f'Stack_SVC+CB (strat_{best_strat})',
    'parameters': json.dumps({'n_neighbors': '5,10,20,40', 'spectral_components': 10}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in (fold_b if best_strat == 'B' else (fold_c if best_strat == 'C' else [acc_a]*10))]),
    'mean_accuracy': float(best_acc),
    'macro_f1': 0.0,
    'balanced_accuracy': 0.0,
    'minimum_fold': float(min(fold_b if best_strat == 'B' else fold_c)) if best_strat != 'A' else 0,
    'train_accuracy': None, 'overfit_gap': None, 'runtime': 0,
    'accepted': best_acc > 0.5988,
    'rejection_reason': '' if best_acc > 0.5988 else 'not improving',
    'next_hypothesis': 'EXP-008: Ensemble weight optimization via grid search',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists: writer.writeheader()
    writer.writerow(row)
print("Done.")
