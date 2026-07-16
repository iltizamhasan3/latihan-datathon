"""
EXP-010: Hierarchical classification.
Level 1: Separate {0,1} vs {2,3} (binary)
Level 2a: Within {0,1} → specialist for classes 0 vs 1
Level 2b: Within {2,3} → specialist for classes 2 vs 3

This leverages that the real boundary may be a "low vs high" grouping.
"""
import numpy as np, pandas as pd, sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].parent))

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
from scipy import stats as sp_stats
import catboost as cb

import warnings; warnings.filterwarnings('ignore')
SEED = 42; EX_ID = "EXP-010"
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

# Binary groups
y_high = (y >= 2).astype(int)
y_class01 = y.copy()  # only for samples in {0,1}
y_class23 = y.copy()  # only for samples in {2,3}

# ================================================================
# Strategy A: Hierarchical 2-level
# Level 1 predicts low(0-1) vs high(2-3)
# Level 2a predicts 0 vs 1 (only low samples)
# Level 2b predicts 2 vs 3 (only high samples)
# ================================================================
print("\n=== Strategy A: Hierarchical Classification ===")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_hier = np.zeros(n, dtype=int)
fold_scores = []

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    X_te = scaler.transform(Xt_vals)
    rs = 42 + fi

    # Level 1: Low(0-1) vs High(2-3)
    # Use separate CV within training fold for level 1
    tr_y_high = y_high[tr]
    val_y_high = y_high[val]

    l1 = cb.CatBoostClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0)
    l1.fit(X_tr, tr_y_high)
    val_l1_probs = l1.predict_proba(X_val)  # [P(low), P(high)]

    # Level 2a: 0 vs 1 (on samples where true label is 0 or 1)
    mask_01_tr = y[tr] < 2
    tr_01_idx = np.where(mask_01_tr)[0]
    X_tr_01 = X_tr[tr_01_idx]
    y_tr_01 = y[tr][tr_01_idx]

    # Level 2b: 2 vs 3
    mask_23_tr = y[tr] >= 2
    tr_23_idx = np.where(mask_23_tr)[0]
    X_tr_23 = X_tr[tr_23_idx]
    y_tr_23 = y[tr][tr_23_idx]

    # Train specialists within each group
    l2_01 = SVC(C=10, gamma='auto', probability=True, random_state=rs)
    l2_01.fit(X_tr_01, y_tr_01)
    l2_23 = SVC(C=10, gamma='auto', probability=True, random_state=rs)
    l2_23.fit(X_tr_23, y_tr_23)

    # Predict validation
    val_l01_probs = l2_01.predict_proba(X_val)
    val_l23_probs = l2_23.predict_proba(X_val)

    # Combine: P(0) = P(low) * P(0|low)
    #          P(1) = P(low) * P(1|low)
    #          P(2) = P(high) * P(2|high)
    #          P(3) = P(high) * P(3|high)
    final_probs = np.zeros((len(val), 4))
    p_low = val_l1_probs[:, 0]
    p_high = val_l1_probs[:, 1]

    final_probs[:, 0] = p_low * val_l01_probs[:, 0]
    final_probs[:, 1] = p_low * val_l01_probs[:, 1]
    final_probs[:, 2] = p_high * val_l23_probs[:, 0]
    final_probs[:, 3] = p_high * val_l23_probs[:, 1]

    oof_hier[val] = np.argmax(final_probs, axis=1)
    fold_scores.append(accuracy_score(y[val], oof_hier[val]))

acc_a = accuracy_score(y, oof_hier)
print(f"  Hierarchical: {acc_a:.4f}, mean={np.mean(fold_scores):.4f}")

# ================================================================
# Strategy B: Level 1 stacking, specialized L2 models
# Level 1 = SVC+CatBoost stack for low/high
# Level 2 = dedicated CatBoost for within-group
# ================================================================
print("\n=== Strategy B: Stacked Hierarchical ===")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_b = np.zeros(n, dtype=int)
fold_b = []
test_probs_b = []

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    X_te = scaler.transform(Xt_vals)
    rs = 42 + fi

    # Level 1 stack
    l1_svc = SVC(C=50, gamma='auto', probability=True, random_state=rs)
    l1_svc.fit(X_tr, y[tr])

    # Get level 1 OOF probabilities via inner CV for stacking (simplified: use same fold)
    # Simple: SVC gives class probs, then we use these for L2
    val_probs_l1 = l1_svc.predict_proba(X_val)

    # Level 2: train specialists on L1 probs + features
    # For each sample, combine SVC class probs with refined specialist
    mask_01_tr = y[tr] < 2
    mask_23_tr = y[tr] >= 2

    if mask_01_tr.sum() > 0:
        l2_01 = cb.CatBoostClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_seed=rs, verbose=0)
        l2_01.fit(X_tr[mask_01_tr], y[tr][mask_01_tr])

    if mask_23_tr.sum() > 0:
        l2_23 = cb.CatBoostClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_seed=rs, verbose=0)
        l2_23.fit(X_tr[mask_23_tr], y[tr][mask_23_tr])

    # Predict validation
    final_probs = np.zeros((len(val), 4))

    # Use SVC L1 probs as base
    for i in range(len(val)):
        p_l1 = val_probs_l1[i]

        # If L1 is confident about {0,1} vs {2,3}, weight specialists more
        if p_l1[:2].sum() > p_l1[2:].sum():
            # Likely low group
            l1_pred = np.argmax(p_l1[:2])
            if mask_01_tr.sum() > 0:
                p_spec = l2_01.predict_proba(X_val[i:i+1])[0]
                # Blend: L1 probs for all classes weighted by confidence + specialist
                blend = 0.3 * p_l1 + 0.7 * np.array([p_spec[0], p_spec[1], p_l1[2], p_l1[3]])
                final_probs[i] = blend
            else:
                final_probs[i] = p_l1
        else:
            if mask_23_tr.sum() > 0:
                p_spec = l2_23.predict_proba(X_val[i:i+1])[0]
                blend = 0.3 * p_l1 + 0.7 * np.array([p_l1[0], p_l1[1], p_spec[0], p_spec[1]])
                final_probs[i] = blend
            else:
                final_probs[i] = p_l1

    oof_b[val] = np.argmax(final_probs, axis=1)
    fold_b.append(accuracy_score(y[val], oof_b[val]))

acc_b = accuracy_score(y, oof_b)
print(f"  Stacked hierarchical: {acc_b:.4f}")

# ================================================================
# Strategy C: Regression to latent + hierarchical threshold
# ================================================================
print("\n=== Strategy C: Regression + Hierarchical Threshold ===")

from sklearn.ensemble import HistGradientBoostingRegressor

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof_c = np.zeros(n, dtype=int)
fold_c = []

for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    rs = 42 + fi

    # Train regression on continuous target
    reg = HistGradientBoostingRegressor(max_iter=500, max_depth=3, learning_rate=0.05, random_state=rs)
    reg.fit(X_tr, y[tr])

    # Predict continuous score
    tr_score = reg.predict(X_tr)
    val_score = reg.predict(X_val)

    # Find 2 thresholds: first split low(0-1) vs high(2-3), then within each
    best_thresh = None
    best_acc = 0

    # Search t1 (boundary between 1 and 2) and boundaries within groups
    for t_mid in np.percentile(tr_score, [30, 35, 40, 45, 50, 55, 60, 65, 70]):
        mask_low = tr_score <= t_mid
        mask_high = tr_score > t_mid

        if mask_low.sum() < 10 or mask_high.sum() < 10:
            continue

        # Within low group, find t_low boundary
        low_scores = tr_score[mask_low]
        low_y = y[tr][mask_low]
        best_t_low = t_mid
        best_low_acc = 0
        for t_low in np.percentile(low_scores, [30, 40, 50, 60, 70]):
            p_low = (low_scores <= t_low).astype(int)
            a = (p_low == (low_y < 1).astype(int)).mean()
            if a > best_low_acc:
                best_low_acc = a
                best_t_low = t_low

        # Within high group, find t_high boundary
        high_scores = tr_score[mask_high]
        high_y = y[tr][mask_high]
        best_t_high = t_mid
        best_high_acc = 0
        for t_high in np.percentile(high_scores, [30, 40, 50, 60, 70]):
            p_high = (high_scores <= t_high).astype(int)
            a = (p_high == (high_y < 3).astype(int)).mean()
            if a > best_high_acc:
                best_high_acc = a
                best_t_high = t_high

        # Apply to training
        preds = np.zeros(len(y[tr]), dtype=int)
        preds[tr_score <= best_t_low] = 0
        preds[(tr_score > best_t_low) & (tr_score <= t_mid)] = 1
        preds[(tr_score > t_mid) & (tr_score <= best_t_high)] = 2
        preds[tr_score > best_t_high] = 3

        a = (preds == y[tr]).mean()
        if a > best_acc:
            best_acc = a
            best_thresh = (best_t_low, t_mid, best_t_high)

    # Apply to validation
    if best_thresh:
        preds = np.zeros(len(val), dtype=int)
        preds[val_score <= best_thresh[0]] = 0
        preds[(val_score > best_thresh[0]) & (val_score <= best_thresh[1])] = 1
        preds[(val_score > best_thresh[1]) & (val_score <= best_thresh[2])] = 2
        preds[val_score > best_thresh[2]] = 3
        oof_c[val] = preds

    fold_c.append(accuracy_score(y[val], oof_c[val]))

acc_c = accuracy_score(y, oof_c)
print(f"  Regression+hier: {acc_c:.4f}")

# ================================================================
# RESULTS
# ================================================================
print(f"\n{'='*50}")
print(f"EXP-010 HIERARCHICAL RESULTS")
print(f"{'='*50}")
print(f"  A (Simple hier):            {acc_a:.4f}")
print(f"  B (Stacked hier):           {acc_b:.4f}")
print(f"  C (Regression+hier thresh): {acc_c:.4f}")
print(f"  Current best:               0.5988")

best_acc = max(acc_a, acc_b, acc_c)
best_strat = ['A','B','C'][np.argmax([acc_a, acc_b, acc_c])]
delta = best_acc - 0.5988
print(f"\n  Best: {best_strat} = {best_acc:.4f} (vs 0.5988: {delta:+.4f})")

import csv
log_path = EXT_DIR / "experiment_log.csv"
log_exists = log_path.exists()
row = {
    'experiment_id': EX_ID, 'parent_id': 'EXP-004',
    'hypothesis': 'Hierarchical classification (low vs high then within-group refinement) beats flat 4-class',
    'feature_family': 'base+sequence',
    'model_family': f'hierarchical (strat_{best_strat})',
    'parameters': json.dumps({'level_1': 'CB/LR', 'level_2': 'SVC/CB', 'threshold_method': 'reg+hier'}),
    'seed': str(SEED),
    'fold_scores': json.dumps([float(f) for f in (fold_b if best_strat == 'B' else (fold_c if best_strat == 'C' else fold_scores))]),
    'mean_accuracy': float(best_acc),
    'macro_f1': 0.0,
    'minimum_fold': float(min(fold_b if best_strat == 'B' else (fold_c if best_strat == 'C' else fold_scores))),
    'train_accuracy': None, 'overfit_gap': None, 'runtime': 0,
    'accepted': best_acc > 0.5988,
    'rejection_reason': '' if best_acc > 0.5988 else 'not improving',
    'next_hypothesis': 'EXP-011: Pseudo-labeling with high-confidence test predictions',
}
with open(log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not log_exists: writer.writeheader()
    writer.writerow(row)
print("Done.")
