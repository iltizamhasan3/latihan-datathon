"""
Phase 3: Data Predictability Measurement.
NN label agreement, class separability, learning curve, Bayes-error proxy.
"""
import numpy as np, pandas as pd, json, sys, warnings, os, time
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from experiments.features import get_all_features
from forensic_experiments.core import load_data, FOR_DIR, RANDOM_SEEDS

train, test, sample = load_data()
y = train['target'].values
X_all = get_all_features(train).fillna(0).replace([np.inf, -np.inf], 0)
X_vals = X_all.values

print(f"Data shape: {X_vals.shape}")
print(f"Target distribution: {np.bincount(y)}")

results = {}

# ================================================================
# A. NEAREST-NEIGHBOR LABEL AGREEMENT
# ================================================================
print("\n" + "="*60)
print("A. NEAREST-NEIGHBOR LABEL AGREEMENT")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vals)

agreement = {}
for k in [1, 3, 5, 10, 20]:
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    # Exclude self
    neighbor_labels = y[indices[:, 1:]]
    self_label = y[:, np.newaxis]
    agreement[k] = np.mean(np.mean(neighbor_labels == self_label, axis=1))

results['neighbor_agreement'] = agreement
print(f"  NN1  label agreement: {agreement[1]:.4f}")
print(f"  NN3  label agreement: {agreement[3]:.4f}")
print(f"  NN5  label agreement: {agreement[5]:.4f}")
print(f"  NN10 label agreement: {agreement[10]:.4f}")
print(f"  NN20 label agreement: {agreement[20]:.4f}")

# Per-class neighbor agreement
class_agreement = {}
for c in range(4):
    mask = y == c
    class_agreement[c] = {}
    for k in [1, 3, 5, 10]:
        neighbor_labels = y[indices[mask][:, 1:k+1]]
        class_agreement[c][k] = float(np.mean(neighbor_labels == c))

results['per_class_agreement'] = class_agreement
print("\n  Per-class NN agreement:")
for c in range(4):
    print(f"    Class {c}: NN1={class_agreement[c][1]:.4f} NN3={class_agreement[c][3]:.4f} "
          f"NN5={class_agreement[c][5]:.4f} NN10={class_agreement[c][10]:.4f}")

# ================================================================
# B. CLASS SEPARABILITY
# ================================================================
print("\n" + "="*60)
print("B. CLASS SEPARABILITY")
print("="*60)

# Fisher score per feature
from sklearn.preprocessing import LabelBinarizer
f_stats, f_pvals = f_classif(X_scaled, y)
mean_f_stat = float(np.mean(f_stats))
max_f_stat = float(np.max(f_stats))
n_significant = int(np.sum(f_pvals < 0.05))
results['fisher_scores'] = {
    'mean_f_stat': mean_f_stat,
    'max_f_stat': max_f_stat,
    'n_significant_features': n_significant,
    'total_features': len(f_stats)
}
print(f"  Mean F-statistic: {mean_f_stat:.4f}")
print(f"  Max F-statistic:  {max_f_stat:.4f}")
print(f"  Significant (p<0.05): {n_significant}/{len(f_stats)}")

# Mutual information
np.random.seed(42)
mi_scores = mutual_info_classif(X_vals, y, random_state=42)
results['mutual_information'] = {
    'mean': float(np.mean(mi_scores)),
    'max': float(np.max(mi_scores)),
    'median': float(np.median(mi_scores)),
    'std': float(np.std(mi_scores))
}
print(f"  MI mean: {np.mean(mi_scores):.6f}")
print(f"  MI max:  {np.max(mi_scores):.6f}")
print(f"  MI median: {np.median(mi_scores):.6f}")

# Silhouette score
from sklearn.metrics import silhouette_score, davies_bouldin_score
sil = float(silhouette_score(X_scaled, y))
db = float(davies_bouldin_score(X_scaled, y))
results['cluster_metrics'] = {'silhouette': sil, 'davies_bouldin': db}
print(f"  Silhouette: {sil:.4f}")
print(f"  Davies-Bouldin: {db:.4f}")

# Within-class vs between-class variance
class_means = np.array([X_scaled[y == c].mean(axis=0) for c in range(4)])
overall_mean = X_scaled.mean(axis=0)

within_var = 0
for c in range(4):
    diff = X_scaled[y == c] - class_means[c]
    within_var += np.sum(diff ** 2)
between_var = np.sum((class_means - overall_mean) ** 2) * np.bincount(y).sum()
ratio = float(between_var / (within_var + 1e-10))

results['variance_ratio'] = {
    'within_class_var': float(within_var),
    'between_class_var': float(between_var),
    'ratio': ratio
}
print(f"  Within-class / Between-class variance ratio: {ratio:.6f}")

# ================================================================
# C. LEARNING CURVE
# ================================================================
print("\n" + "="*60)
print("C. LEARNING CURVE")
print("="*60)

from sklearn.model_selection import learning_curve as sk_learning_curve

def learning_curve_eval(model_fn, name, model_label):
    """Evaluate model with varying training set sizes."""
    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_lc = {'train_sizes': [], 'train_scores': [], 'val_scores': []}

    for size in train_sizes:
        if size < 1.0:
            n_train = int(len(y) * size)
            # Stratified subsample
            from sklearn.model_selection import train_test_split
            _, idx_sub = train_test_split(np.arange(len(y)), train_size=size,
                                          stratify=y, random_state=42)
            X_sub = X_vals[idx_sub]
            y_sub = y[idx_sub]
        else:
            X_sub = X_vals
            y_sub = y

        fold_scores = []
        for fi, (tr, val) in enumerate(cv.split(X_sub, y_sub)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_sub[tr])
            X_val = scaler.transform(X_sub[val])
            m = model_fn(42 + fi)
            m.fit(X_tr, y_sub[tr])
            val_acc = accuracy_score(y_sub[val], m.predict(X_val))
            fold_scores.append(val_acc)

        results_lc['train_sizes'].append(size * len(y))
        results_lc['val_scores'].append(float(np.mean(fold_scores)))
        print(f"    {model_label:20s} {int(size*100):3d}% data ({int(size*len(y)):4d} rows): "
              f"val={np.mean(fold_scores):.4f}")

    return results_lc

# Test with SVC (fast)
svc_lc = learning_curve_eval(lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
                              'SVC_C50', 'SVC')
results['learning_curve_svc'] = svc_lc

# Test with RF (fast)
rf_lc = learning_curve_eval(
    lambda s: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=s, n_jobs=-1),
    'RF', 'RF')
results['learning_curve_rf'] = rf_lc

# ================================================================
# D. BAYES-ERROR PROXY
# ================================================================
print("\n" + "="*60)
print("D. BAYES-ERROR PROXY")
print("="*60)

# KNN error as baseline
knn_errors = {}
for k in [1, 3, 5, 7, 11, 15, 21]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    oof = np.zeros(len(y), dtype=int)
    for tr, val in cv.split(X_vals, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_vals[tr])
        X_val = scaler.transform(X_vals[val])
        m = KNeighborsClassifier(n_neighbors=k, weights='distance')
        m.fit(X_tr, y[tr])
        oof[val] = m.predict(X_val)
    err = 1 - accuracy_score(y, oof)
    knn_errors[k] = float(err)
    print(f"  KNN(k={k:2d}) error= {err:.4f}")

results['knn_bayes_proxy'] = knn_errors
best_knn_err = min(knn_errors.values())
results['empirical_bayes_estimate'] = float(best_knn_err)
print(f"\n  Best KNN error: {best_knn_err:.4f}")
print(f"  Bayes-error proxy (lower bound): ~{best_knn_err:.4f}")

# Ensemble disagreement
print("\n  Ensemble disagreement (model diversity):")
models_to_check = [
    ('SVC_C50', lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s)),
    ('RF_200', lambda s: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=s, n_jobs=-1)),
    ('HGB_300', lambda s: HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=s)),
]

model_preds = {}
for mname, mfn in models_to_check:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    oof = np.zeros(len(y), dtype=int)
    for tr, val in cv.split(X_vals, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_vals[tr])
        X_val = scaler.transform(X_vals[val])
        m = mfn(42)
        m.fit(X_tr, y[tr])
        oof[val] = m.predict(X_val)
    model_preds[mname] = oof
    acc = accuracy_score(y, oof)
    print(f"    {mname:15s} acc={acc:.4f}")

# Pairwise disagreement
n_models = len(models_to_check)
model_names = list(model_preds.keys())
pair_disagreement = 0
pair_count = 0
all_disagree = np.zeros(len(y), dtype=float)
for i in range(n_models):
    for j in range(i+1, n_models):
        disagree = np.mean(model_preds[model_names[i]] != model_preds[model_names[j]])
        pair_disagreement += disagree
        pair_count += 1
        all_disagree += (model_preds[model_names[i]] != model_preds[model_names[j]]).astype(float)

mean_disagreement = float(pair_disagreement / pair_count)
results['ensemble_disagreement'] = mean_disagreement
print(f"\n  Mean pairwise model disagreement: {mean_disagreement:.4f}")

# Samples where all models are wrong
all_wrong = np.ones(len(y), dtype=bool)
for mname in model_names:
    all_wrong &= (model_preds[mname] != y)
n_all_wrong = int(np.sum(all_wrong))
results['all_models_wrong_count'] = n_all_wrong
results['all_models_wrong_pct'] = float(n_all_wrong / len(y) * 100)
print(f"  Samples where ALL models wrong: {n_all_wrong}/{len(y)} ({n_all_wrong/len(y)*100:.1f}%)")

# Samples where majority is wrong
majority_wrong = np.zeros(len(y), dtype=int)
for mname in model_names:
    majority_wrong += (model_preds[mname] != y).astype(int)
n_majority_wrong = int(np.sum(majority_wrong >= 2))
results['majority_wrong_count'] = n_majority_wrong
results['majority_wrong_pct'] = float(n_majority_wrong / len(y) * 100)
print(f"  Samples where >= 2/3 models wrong: {n_majority_wrong}/{len(y)} ({n_majority_wrong/len(y)*100:.1f}%)")

# ================================================================
# SAVE RESULTS
# ================================================================
output = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_shape': X_vals.shape,
    'class_distribution': np.bincount(y).tolist(),
    'results': results
}

with open(FOR_DIR / "feature_analysis" / "predictability.json", 'w') as f:
    json.dump({k: v for k, v in output.items() if k != 'results' or True}, f, indent=2, default=str)

# Save detailed per-sample neighbor agreement for label quality analysis
sample_nn_agreement = np.mean(y[indices[:, 1:4]] == y[:, np.newaxis], axis=1)  # NN3 agreement
nn_df = pd.DataFrame({
    'id': np.arange(len(y)),
    'target': y,
    'nn1_agreement': (y[indices[:, 1]] == y).astype(int),
    'nn3_agreement': np.mean(y[indices[:, 1:4]] == y[:, np.newaxis], axis=1),
    'nn5_agreement': np.mean(y[indices[:, 1:6]] == y[:, np.newaxis], axis=1),
    'nn10_agreement': np.mean(y[indices[:, 1:11]] == y[:, np.newaxis], axis=1),
})
nn_df.to_csv(FOR_DIR / "label_analysis" / "neighbor_agreement.csv", index=False)
print(f"\nSaved neighbor_agreement.csv ({len(nn_df)} samples)")

print(f"\n{'='*60}")
print("PREDICTABILITY ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Best KNN error: {best_knn_err:.4f} -> acc={1-best_knn_err:.4f}")
print(f"All models wrong: {n_all_wrong} samples ({n_all_wrong/len(y)*100:.1f}%)")
print(f"Majority(2/3) wrong: {n_majority_wrong} samples ({n_majority_wrong/len(y)*100:.1f}%)")
print(f"Estimated realistic ceiling: ~{1-best_knn_err:.3f} (KNN lower bound)")
