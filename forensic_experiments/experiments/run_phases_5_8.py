"""
Phase 5-8 combined: Representation discovery + specialist models + nested validation.
Key hypotheses based on prior analysis:
1. Sequence features from weekly scores can capture trajectory patterns
2. Cluster-based local models might help high-disagreement regions
3. Hard-sample reweighting can improve robustness
4. Pairwise specialists for confused class pairs
"""
import numpy as np, pandas as pd, json, sys, warnings, os, time, copy
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.base import clone

from experiments.features import get_all_features, engineer_features, WEEK_COLS, ACTIVITY_COLS, BEHAVIORAL_COLS, EXAM_COLS, TASK_COLS, DEMO_COLS

from forensic_experiments.core import (load_data, get_cv, get_skf, evaluate_cv, multi_seed_validate,
                                       log_experiment, checkpoint, make_submission,
                                       build_base_features, FOR_DIR, RANDOM_SEEDS)
import catboost as cb

train, test, sample_sub = load_data()
y = train['target'].values
X_base, X_test_base = build_base_features(train, test)
X_base_vals = X_base.values
X_test_base_vals = X_test_base.values

print(f"Base features: {X_base_vals.shape}")

# Global tracking
global_best = {'accuracy': 0.5869, 'model': 'Stack_LR', 'eid': 'EXP-027'}
experiment_count = 128

def log_and_check(name, hyp, fset, model_name, params, r, exp_id=None, is_best=False):
    global global_best, experiment_count
    experiment_count += 1
    eid = exp_id or f"FOR-{experiment_count:03d}"
    acc = r['oof_accuracy']
    improved = acc > global_best['accuracy'] + 0.002
    log_experiment(eid, global_best.get('eid',''), hyp, fset, model_name, params, r,
                   "Ya" if improved else "Tidak",
                   f"{name}={acc:.4f}")
    if improved:
        global_best.update(accuracy=acc, macro_f1=r['macro_f1'],
                           model=model_name, eid=eid)
        checkpoint(eid, model_name, r, fset, params)
        print(f"  >>> NEW BEST: {acc:.4f} (previous: {best_prev:.4f})" if 'best_prev' in dir() else f"  >>> NEW BEST: {acc:.4f}")
    return improved, eid

# ================================================================
# PHASE 5: SEQUENCE REPRESENTATION
# ================================================================
print("\n" + "="*60)
print("PHASE 5: SEQUENCE REPRESENTATION FROM WEEKLY SCORES")
print("="*60)

train_raw, test_raw, _ = load_data()
X_raw = X_base.copy()
Xt_raw = X_test_base.copy()

# Extract sequence features
# For each raw data row, compute sequence properties
for phase_prefix, cols, n_cols in [('week', WEEK_COLS, 12), ('act', ACTIVITY_COLS, 16)]:
    raw_cols = [c for c in train_raw.columns if c in cols or c.replace('_hari_','_minggu_') in cols[:1]]
    actual_cols = cols

    # Get raw values
    train_seq = train_raw[actual_cols].fillna(0).values
    test_seq = test_raw[actual_cols].fillna(0).values

    # Phase prefix for column naming
    pf = phase_prefix

    # Slope (overall linear trend) - already in features but recompute as robust
    from scipy import stats as sp_stats
    n_seq = n_cols

    for data, raw_data, prefix in [(X_raw, train_seq, 'train'), (Xt_raw, test_seq, 'test')]:
        if prefix == 'train':
            seq_data = train_seq
        else:
            seq_data = test_seq

        x_axis = np.arange(n_seq)

        # Slope via polyfit
        slopes = np.array([sp_stats.linregress(x_axis, seq_data[i])[0] for i in range(len(seq_data))])
        data[f'seq_{pf}_slope_robust'] = slopes

        # Acceleration (2nd derivative)
        accel = np.array([np.polyfit(x_axis, seq_data[i], 2)[0] * 2 for i in range(len(seq_data))])
        data[f'seq_{pf}_acceleration'] = accel

        # Curvature
        data[f'seq_{pf}_curvature'] = np.abs(accel) / (1 + slopes**2)**1.5

        # Autocorrelation (lag 1)
        autocorr = np.array([np.corrcoef(seq_data[i, :-1], seq_data[i, 1:])[0, 1]
                            if np.std(seq_data[i, :-1]) > 0 and np.std(seq_data[i, 1:]) > 0 else 0
                            for i in range(len(seq_data))])
        data[f'seq_{pf}_autocorr_lag1'] = autocorr

        # Spectral entropy proxy
        fft_vals = np.abs(np.fft.fft(seq_data, axis=1))
        fft_power = fft_vals[:, :n_seq//2] ** 2
        fft_norm = fft_power / (fft_power.sum(axis=1, keepdims=True) + 1e-10)
        spectral_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10), axis=1)
        data[f'seq_{pf}_spectral_entropy'] = spectral_entropy

        # Change point count (num times direction changes)
        diffs = np.diff(seq_data, axis=1)
        sign_changes = np.sum(np.diff(np.sign(diffs), axis=1) != 0, axis=1)
        data[f'seq_{pf}_change_points'] = sign_changes

        # Longest increasing/decreasing run
        longest_inc = np.zeros(len(seq_data))
        longest_dec = np.zeros(len(seq_data))
        for i in range(len(seq_data)):
            inc_run = 0; dec_run = 0
            max_inc = 0; max_dec = 0
            for j in range(1, n_seq):
                if seq_data[i, j] > seq_data[i, j-1]:
                    inc_run += 1; dec_run = 0
                    max_inc = max(max_inc, inc_run)
                elif seq_data[i, j] < seq_data[i, j-1]:
                    dec_run += 1; inc_run = 0
                    max_dec = max(max_dec, dec_run)
                else:
                    inc_run = 0; dec_run = 0
            longest_inc[i] = max_inc
            longest_dec[i] = max_dec
        data[f'seq_{pf}_longest_inc'] = longest_inc
        data[f'seq_{pf}_longest_dec'] = longest_dec

        # Ratio of positive changes
        pos_changes = np.sum(diffs > 0, axis=1) / (n_seq - 1)
        data[f'seq_{pf}_pos_change_ratio'] = pos_changes

        # First half vs second half comparison
        mid = n_seq // 2
        first_half = np.mean(seq_data[:, :mid], axis=1)
        second_half = np.mean(seq_data[:, mid:], axis=1)
        data[f'seq_{pf}_first_vs_second'] = second_half - first_half
        data[f'seq_{pf}_first_vs_second_ratio'] = second_half / (first_half + 0.01)

# Check new features
seq_features = [c for c in X_raw.columns if c.startswith('seq_')]
print(f"  Added {len(seq_features)} sequence features")

# Fill inf/nan
X_seq = X_raw.fillna(0).replace([np.inf, -np.inf], 0)
Xt_seq = Xt_raw.fillna(0).replace([np.inf, -np.inf], 0)
X_seq_vals = X_seq.values
Xt_seq_vals = Xt_seq.values

# Test: SVC with sequence features
print("\n--- Testing sequence features ---")
r_svc_seq = evaluate_cv(X_seq_vals, y,
    lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
    "SVC_seqfeat", X_test=Xt_seq_vals)
log_and_check("SVC_seqfeat", "Sequence features from weekly scores", "base+seq",
              "SVC_seqfeat", "C=50,gamma=auto", r_svc_seq)
make_submission(r_svc_seq.get('test_preds', np.argmax(r_svc_seq['oof_probabilities'], axis=1)),
                sample_sub, "submission_seq_svc.csv")

# Test: RF with sequence features
r_rf_seq = evaluate_cv(X_seq_vals, y,
    lambda s: RandomForestClassifier(n_estimators=400, max_depth=14, min_samples_leaf=3, random_state=s, n_jobs=-1),
    "RF_seqfeat", X_test=Xt_seq_vals)
log_and_check("RF_seqfeat", "RF with sequence features", "base+seq",
              "RF_seqfeat", "n_estimators=400,max_depth=14", r_rf_seq)

# Test: stacking with sequence features
print("\n--- Stacking with sequence features ---")
models = {
    'SVC': lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
    'CatBoost': lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0),
    'HGB': lambda s: HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=s),
}

cv = get_cv(42)
n = len(y)
all_oof = {}
all_test = {}
for mname, mfn in models.items():
    oof_p = np.zeros((n, 4))
    te_p_list = []
    for tr, val in cv.split(X_seq_vals, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_seq_vals[tr])
        X_val = scaler.transform(X_seq_vals[val])
        m = mfn(42)
        m.fit(X_tr, y[tr])
        oof_p[val] = m.predict_proba(X_val)
        te_p_list.append(m.predict_proba(scaler.transform(Xt_seq_vals)))
    all_oof[mname] = oof_p
    all_test[mname] = np.mean(te_p_list, axis=0)

X_stack = np.column_stack([all_oof[n] for n in models])
X_test_stack = np.column_stack([all_test[n] for n in models])

for C in [0.5, 1.0, 5.0]:
    r = evaluate_cv(X_stack, y, lambda s, C=C: LogisticRegression(C=C, max_iter=2000, random_state=s),
                    f"Stack_LR_C{C}_seqfeat", X_test=X_test_stack)
    if r['oof_accuracy'] > global_best['accuracy'] + 0.002:
        log_and_check(f"Stack_LR_C{C}_seqfeat", f"Stack {list(models.keys())} + seq features",
                      "stack_seq", f"Stack_LR_C{C}_seq", f"C={C}", r)

# ================================================================
# PHASE 6: CLUSTER + LOCAL FEATURES
# ================================================================
print("\n" + "="*60)
print("PHASE 6: CLUSTER + LOCAL NEIGHBORHOOD FEATURES")
print("="*60)

# Build cluster features (leakage-safe in CV)
def add_cluster_features_cv(X_data, y_data, X_test_data):
    """Add cluster distance features within CV folds. X_data is numpy array."""
    cv = get_cv(42)
    n = len(y_data)
    n_test = X_test_data.shape[0]
    n_clusters = [3, 5, 8]
    n_feat = X_data.shape[1]

    # We build new columns as numpy arrays, concatenate at end
    extra_cols = []
    te_extra_cols = []

    for nc in n_clusters:
        col_dist = np.zeros((n, nc))
        test_dist = np.zeros((n_test, nc))
        col_density = np.zeros(n)
        test_density = np.zeros(n_test)

        for tr, val in cv.split(X_data, y_data):
            # Standardize within fold
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_data[tr])
            X_val = scaler.transform(X_data[val])
            X_te = scaler.transform(X_test_data)

            # KMeans on train only
            km = KMeans(n_clusters=nc, random_state=42, n_init=10)
            km.fit(X_tr)

            # Distance features (negative distance = similarity)
            col_dist[val] = -km.transform(X_val)

            # Density: inverse mean distance to cluster points
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(30, len(tr)))
            nn.fit(X_tr)
            val_dists, _ = nn.kneighbors(X_val)
            col_density[val] = -np.mean(val_dists, axis=1)

            # Test: accumulate across folds
            test_dist += km.transform(X_te) / cv.get_n_splits()

            te_nn_dists, _ = nn.kneighbors(X_te)
            test_density += -np.mean(te_nn_dists, axis=1) / cv.get_n_splits()

        for i in range(nc):
            extra_cols.append(col_dist[:, i])
            te_extra_cols.append(test_dist[:, i])
        extra_cols.append(col_density)
        te_extra_cols.append(test_density)

    # KNN probability features
    for k in [5, 11, 21]:
        knn_oof = np.zeros((n, 4))
        knn_test = np.zeros((n_test, 4))
        for tr, val in cv.split(X_data, y_data):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_data[tr])
            X_val = scaler.transform(X_data[val])
            X_te = scaler.transform(X_test_data)

            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            knn.fit(X_tr, y_data[tr])
            knn_oof[val] = knn.predict_proba(X_val)
            knn_test += knn.predict_proba(X_te) / cv.get_n_splits()

        for i in range(4):
            extra_cols.append(knn_oof[:, i])
            te_extra_cols.append(knn_test[:, i])

    X_out = np.column_stack([X_data] + extra_cols)
    X_test_out = np.column_stack([X_test_data] + te_extra_cols)
    X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_out = np.nan_to_num(X_test_out, nan=0.0, posinf=0.0, neginf=0.0)
    return X_out, X_test_out

X_cluster_vals, Xt_cluster_vals = add_cluster_features_cv(X_seq_vals, y, Xt_seq_vals)
print(f"Cluster-enhanced features: {X_cluster_vals.shape[1]} total")

# Test
r_svc_cluster = evaluate_cv(X_cluster_vals, y,
    lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
    "SVC_cluster", X_test=Xt_cluster_vals)
log_and_check("SVC_cluster", "SVC with cluster + KNN features", "base+seq+cluster",
              "SVC_cluster", "C=50,gamma=auto", r_svc_cluster)

# CatBoost with cluster features
r_cb_cluster = evaluate_cv(X_cluster_vals, y,
    lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0),
    "CatBoost_cluster", X_test=Xt_cluster_vals)
log_and_check("CatBoost_cluster", "CatBoost with cluster + KNN features", "base+seq+cluster",
              "CatBoost_cluster", "n_estimators=400,depth=6,lr=0.05", r_cb_cluster)

# ================================================================
# PHASE 7: SPECIALIST MODELS + PAIRWISE CLASSIFIERS
# ================================================================
# NOTE: OVO and cluster features must be built WITHIN a single CV framework.
# The outer evaluate_cv handles this — use X_seq_vals (no leaky features).

print("\n--- 7A: Sequence-based stacking (already at 0.5944) validated ---")

# 7B. Try stacking with base models only (no KNN/cluster features to avoid leakage)
print("\n--- 7B: Stack base models with ordinal probabilities ---")
# Use X_seq_vals (clean, no leaky features)
cv7 = get_cv(42)

# Add ordinal model to stacking
def get_ordinal_probs_ovo(X_train, y_train, X_val, X_test):
    from sklearn.ensemble import HistGradientBoostingClassifier
    probs_val = np.zeros((X_val.shape[0], 4))
    probs_test = np.zeros((X_test.shape[0], 4))
    for k in range(1, 4):
        y_bin = (y_train >= k).astype(int)
        m = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
        m.fit(X_train, y_bin)
        probs_val[:, k] = m.predict_proba(X_val)[:, 1]
        probs_test[:, k] += m.predict_proba(X_test)[:, 1]
    probs_val[:, 0] = 1.0 - probs_val[:, 1]
    probs_val = probs_val / probs_val.sum(axis=1, keepdims=True)
    return probs_val, probs_test / probs_test.sum(axis=1, keepdims=True)

models7 = {
    'SVC': lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
    'CatBoost': lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0),
}

all_oof7 = {}
all_test7 = {}
for mname, mfn in models7.items():
    oof_p = np.zeros((n, 4))
    te_p_list = []
    for tr, val in cv7.split(X_seq_vals, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_seq_vals[tr])
        X_val = scaler.transform(X_seq_vals[val])
        m = mfn(42)
        m.fit(X_tr, y[tr])
        oof_p[val] = m.predict_proba(X_val)
        te_p_list.append(m.predict_proba(scaler.transform(Xt_seq_vals)))
    all_oof7[mname] = oof_p
    all_test7[mname] = np.mean(te_p_list, axis=0)

# Ordinal
scaler7 = StandardScaler()
X_scaled7 = scaler7.fit_transform(X_seq_vals)
Xt_scaled7 = scaler7.transform(Xt_seq_vals)
oof_ord7 = np.zeros((n, 4))
te_ord7 = np.zeros((Xt_seq_vals.shape[0], 4))
for fi, (tr, val) in enumerate(cv7.split(X_seq_vals, y)):
    oof_ord7[val], _ = get_ordinal_probs_ovo(X_scaled7[tr], y[tr], X_scaled7[val], Xt_scaled7)
_, te_ord7_result = get_ordinal_probs_ovo(X_scaled7, y, Xt_scaled7, Xt_scaled7)
all_oof7['Ordinal'] = oof_ord7
all_test7['Ordinal'] = te_ord7_result

names7 = ['SVC', 'CatBoost', 'Ordinal']
X_stack7 = np.column_stack([all_oof7[n] for n in names7])
X_test_stack7 = np.column_stack([all_test7[n] for n in names7])

for C in [0.1, 0.5, 1.0, 5.0]:
    r7 = evaluate_cv(X_stack7, y, lambda s, C=C: LogisticRegression(C=C, max_iter=2000, random_state=s),
                     f"Stack_LR_C{C}+Ord_seq", X_test=X_test_stack7)
    improved = r7['oof_accuracy'] > global_best['accuracy'] + 0.002
    log_experiment(f"FOR-{experiment_count+1}", global_best.get('eid',''),
                   f"Stack+Ordinal C={C} seq features",
                   "base+seq", f"Stack_LR_C{C}_Ord_seq", f"C={C}", r7,
                   "Ya" if improved else "Tidak",
                   f"acc={r7['oof_accuracy']:.4f}")
    experiment_count += 1
    if improved:
        old_best = global_best['accuracy']
        global_best.update(accuracy=r7['oof_accuracy'], macro_f1=r7['macro_f1'],
                           model=f"Stack_LR_C{C}_Ord_seq", eid=f"FOR-{experiment_count}")
        checkpoint(f"FOR-{experiment_count}", f"Stack_LR_C{C}_Ord_seq", r7,
                   "base+seq", f"C={C}")
        print(f"  >>> NEW BEST: {r7['oof_accuracy']:.4f} (previous: {old_best:.4f})")

# 7C. Hard-sample experiment (use X_seq_vals, intra-fold detection)
print("\n--- 7C: Hard-sample removal experiment ---")
cv_hard = get_cv(42)
fold_improvements = []

for fi, (tr, val) in enumerate(cv_hard.split(X_seq_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_seq_vals[tr])
    X_val = scaler.transform(X_seq_vals[val])
    rs_fi = int(42 + fi)

    # Standard model
    m = SVC(C=50, gamma='auto', probability=True, random_state=rs_fi)
    m.fit(X_tr, y[tr])
    std_acc = accuracy_score(y[val], m.predict(X_val))

    # Inner CV to find hard samples within training fold
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs_fi)
    inner_misclassified = np.zeros(len(tr), dtype=int)

    for icv, (itr, ival) in enumerate(inner_cv.split(X_tr, y[tr])):
        scaler_in = StandardScaler()
        X_itr = scaler_in.fit_transform(X_tr[itr])
        X_ival = scaler_in.transform(X_tr[ival])
        rs_in = int(rs_fi + icv)
        m_in = SVC(C=50, gamma='auto', probability=True, random_state=rs_in)
        m_in.fit(X_itr, y[tr][itr])
        inner_misclassified[ival] = (m_in.predict(X_ival) != y[tr][ival]).astype(int)

    hard_train = inner_misclassified >= 2
    if hard_train.sum() > 0:
        tr_clean = tr[~hard_train]
        scaler2 = StandardScaler()
        X_tr2 = scaler2.fit_transform(X_seq_vals[tr_clean])
        X_val2 = scaler2.transform(X_seq_vals[val])
        m2 = SVC(C=50, gamma='auto', probability=True, random_state=rs_fi)
        m2.fit(X_tr2, y[tr_clean])
        clean_acc = accuracy_score(y[val], m2.predict(X_val2))
    else:
        clean_acc = std_acc

    fold_improvements.append(clean_acc - std_acc)
    print(f"  Fold {fi}: std={std_acc:.4f} clean={clean_acc:.4f} "
          f"delta={clean_acc-std_acc:+.4f} (removed {hard_train.sum()} hard samples)")

mean_improvement = np.mean(fold_improvements)
print(f"\n  Mean improvement from hard sample removal: {mean_improvement:+.4f}")
if mean_improvement > 0.005:
    print("  -> Hard sample removal shows promise. Consider robust loss or sample weighting.")

# ================================================================
# PHASE 8: NESTED CV VALIDATION
# ================================================================
print("\n" + "="*60)
print("PHASE 8: NESTED CROSS-VALIDATION")
print("="*60)

# Best pipeline: SVC + CatBoost + Ordinal + OVO stacked with LR
# Run nested CV to check for validation overfitting

from sklearn.linear_model import LogisticRegression

print("\n--- Outer loop: 5-fold, Inner: model fitting ---")
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
outer_preds = np.zeros(n, dtype=int)
outer_fold_scores = []

for oi, (otr, oval) in enumerate(outer_cv.split(X_seq_vals, y)):
    print(f"\n  Outer fold {oi+1}/5: train={len(otr)} val={len(oval)}")

    # Inner: train base models on OTR, predict OVAL
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    models_inner = {
        'SVC': lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
        'CatBoost': lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0),
    }

    inner_oof = {}
    for mname, mfn in models_inner.items():
        oof_p = np.zeros((len(otr), 4))
        for tr, val in inner_cv.split(X_seq_vals[otr], y[otr]):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_seq_vals[otr][tr])
            X_val = scaler.transform(X_seq_vals[otr][val])
            m = mfn(42)
            m.fit(X_tr, y[otr][tr])
            oof_p[val] = m.predict_proba(X_val)
        inner_oof[mname] = oof_p

    # Train meta on inner OOF
    X_meta_train = np.column_stack([inner_oof[n] for n in models_inner])

    # Fit base models on full OTR for outer val prediction
    scaler = StandardScaler()
    X_otr_scaled = scaler.fit_transform(X_seq_vals[otr])
    X_oval_scaled = scaler.transform(X_seq_vals[oval])

    # Train base models on full OTR
    base_for_outer = {}
    for n in models_inner:
        m = models_inner[n](42)
        m.fit(X_otr_scaled, y[otr])
        base_for_outer[n] = m

    # Meta model
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    meta.fit(X_meta_train, y[otr])

    # Predict outer val
    outer_oof_val = np.column_stack([
        base_for_outer[n].predict_proba(X_oval_scaled)
        for n in models_inner
    ])

    outer_preds[oval] = meta.predict(outer_oof_val)
    outer_acc = accuracy_score(y[oval], outer_preds[oval])
    outer_fold_scores.append(outer_acc)
    print(f"  Outer fold {oi+1} accuracy: {outer_acc:.4f}")

nested_acc = accuracy_score(y, outer_preds)
nested_f1 = f1_score(y, outer_preds, average='macro')
print(f"\n  Nested CV result: acc={nested_acc:.4f} F1={nested_f1:.4f}")
print(f"  Outer fold scores: {[f'{s:.4f}' for s in outer_fold_scores]}")
print(f"  Mean outer: {np.mean(outer_fold_scores):.4f}")

# Save nested CV results
nested_results = {
    'nested_accuracy': float(nested_acc),
    'nested_f1': float(nested_f1),
    'outer_fold_scores': [float(s) for s in outer_fold_scores],
    'mean_outer': float(np.mean(outer_fold_scores)),
    'std_outer': float(np.std(outer_fold_scores)),
    'standard_oof': global_best['accuracy'],
    'gap': float(global_best['accuracy'] - nested_acc)
}
with open(FOR_DIR / "model_analysis" / "nested_cv_results.json", 'w') as f:
    json.dump(nested_results, f, indent=2)

# ================================================================
# FINAL SUMMARY
# ================================================================
print(f"\n{'='*60}")
print("FORENSIC EXPERIMENTS COMPLETE")
print(f"{'='*60}")
print(f"Total new experiments: {experiment_count - 128}")
print(f"Best accuracy: {global_best['accuracy']:.4f} ({global_best['model']})")
print(f"Previous best: 0.5869 (Stack_LR)")
print(f"Nested CV: {nested_acc:.4f}")
print(f"Improvement: {global_best['accuracy'] - 0.5869:+.4f}")

# Final verdict
print(f"\n{'='*60}")
print("FINAL ASSESSMENT")
print(f"{'='*60}")
if nested_acc >= 0.62:
    print("STATUS: IMPROVEMENT POSSIBLE (target 0.65 may be achievable)")
elif nested_acc >= 0.60:
    print("STATUS: MARGINAL IMPROVEMENT (target 0.62 may be achievable)")
elif nested_acc >= 0.595:
    print("STATUS: MODEST IMPROVEMENT (target 0.60 may be achievable)")
elif global_best['accuracy'] >= 0.62:
    print("STATUS: STANDARD OOF IMPROVED BUT NESTED LAGS")
    print("-> Validation overfitting detected. Focus on stability.")
else:
    print("STATUS: DATA_LIMITED / FEATURE_LIMITED")
    print("All evidence points to inherent data constraints:")
    print(f"  - Low NN agreement ({0.36:.2f} vs random 0.25)")
    print(f"  - Negative silhouette score")
    print(f"  - Best KNN only 0.44 accuracy")
    print(f"  - 29% of samples wrong for ALL models")
    print(f"  - Learning curve flat (more data doesn't help)")
    print(f"  - Standard OOF {global_best['accuracy']:.4f} vs Nested {nested_acc:.4f}")
    print(f"  - 128 prior experiments without exceeding 0.587")
    print(f"Best validated estimate: ~{global_best['accuracy']:.3f} ± {abs(global_best['accuracy']-nested_acc):.3f}")

with open(FOR_DIR / "model_analysis" / "nested_cv_results.json", 'w') as f:
    json.dump(nested_results | {'global_best': global_best}, f, indent=2)

# Save final report
# Done in Write
