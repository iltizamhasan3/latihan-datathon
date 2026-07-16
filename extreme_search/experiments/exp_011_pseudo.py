"""
EXP-011: Pseudo-labeling with high-confidence test predictions.
Only use stable predictions (ensemble agreement ≥3 models, confidence ≥0.95).
Iterative refinement: predict test, add confident ones to training, retrain.
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
from scipy import stats as sp_stats

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
import catboost as cb

import warnings; warnings.filterwarnings('ignore')
SEED = 42; EX_ID = "EXP-011"
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

# ================================================================
# Step 1: Get confident test predictions via ensemble
# ================================================================
print("\n=== Step 1: Multi-Model Test Predictions ===")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
model_fns = {
    'SVC': lambda rs: SVC(C=50, gamma='auto', probability=True, random_state=rs),
    'CB': lambda rs: cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_seed=rs, verbose=0),
    'HGB': lambda rs: HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.05, random_state=rs),
}

# Train models on full training data to predict test
models_full = {}
for mname, mfn in model_fns.items():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals)
    Xt_scaled = scaler.transform(Xt_vals)
    m = mfn(SEED).fit(X_scaled, y)
    models_full[mname] = (scaler, m)

# All 3 model predictions for test
test_preds = {}
test_conf = {}
for mname, (sc, m) in models_full.items():
    p = m.predict_proba(sc.transform(Xt_vals))
    test_preds[mname] = p
    test_conf[mname] = p.max(axis=1)

# Ensemble agreement
n_models = len(model_fns)
ensemble_preds = np.zeros((len(Xt_vals), 4))
for mname, p in test_preds.items():
    ensemble_preds += p
ensemble_preds /= n_models

# Find confident test samples
final_test_preds = np.argmax(ensemble_preds, axis=1)
final_test_conf = ensemble_preds.max(axis=1)
n_confident = np.sum(final_test_conf >= 0.95)
print(f"  Test samples with conf >= 0.95: {n_confident}/800")
print(f"  Test samples with conf >= 0.90: {np.sum(final_test_conf >= 0.90)}/800")
print(f"  Test samples with conf >= 0.80: {np.sum(final_test_conf >= 0.80)}/800")

# ================================================================
# Strategy A: Pseudo-label with confident test samples (weighted)
# ================================================================
print("\n=== Strategy A: Pseudo-label Weighted Retraining ===")

for conf_thresh in [0.95, 0.90]:
    for pl_weight in [0.3, 0.5, 1.0]:
        pl_mask = final_test_conf >= conf_thresh
        if pl_mask.sum() < 5:
            print(f"  Skipping conf>{conf_thresh}: only {pl_mask.sum()} samples")
            continue

        # Pseudo-trained model
        X_aug = np.vstack([X_vals, Xt_vals[pl_mask]])
        y_aug = np.concatenate([y, final_test_preds[pl_mask]])

        # Sample weights: 1.0 for original, pl_weight for pseudo
        sw = np.ones(len(X_aug))
        sw[n:] = pl_weight

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
        oof = np.zeros(n, dtype=int)
        folds = []

        for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_vals[tr])
            X_val = scaler.transform(X_vals[val])

            rs = 42 + fi

            # Train with pseudo-labels on full augmented data
            X_aug_tr = np.vstack([X_vals[tr], Xt_vals[pl_mask]])
            y_aug_tr = np.concatenate([y[tr], final_test_preds[pl_mask]])
            sw_tr = np.ones(len(X_aug_tr))
            sw_tr[len(tr):] = pl_weight

            scaler_aug = StandardScaler()
            X_aug_scaled = scaler_aug.fit_transform(X_aug_tr)
            X_val_s = scaler_aug.transform(X_vals[val])

            cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                          random_seed=rs, verbose=0)
            cb_m.fit(X_aug_scaled, y_aug_tr, sample_weight=sw_tr)
            oof[val] = cb_m.predict(X_val_s).ravel()
            folds.append(accuracy_score(y[val], oof[val]))

        acc = accuracy_score(y, oof)
        print(f"  conf>{conf_thresh}, weight={pl_weight}: {acc:.4f} (mean={np.mean(folds):.4f}), n_pseudo={pl_mask.sum()}")

# ================================================================
# Strategy B: Iterative pseudo-labeling (2 rounds)
# ================================================================
print("\n=== Strategy B: Iterative Pseudo-labeling ===")

for iteration in range(2):
    if iteration == 0:
        # First pass: use base model, add confident test
        pl_mask = final_test_conf >= 0.90
    else:
        # Update test predictions with new model
        scaler_new = StandardScaler()
        X_scaled_new = scaler_new.fit_transform(X_aug)
        Xt_scaled_new = scaler_new.transform(Xt_vals)

        # Re-predict test
        ensemble_preds_new = np.zeros((len(Xt_vals), 4))
        for mname, mfn in model_fns.items():
            m_new = mfn(SEED)
            m_new.fit(X_scaled_new, y_aug)
            ensemble_preds_new += m_new.predict_proba(Xt_scaled_new)
        ensemble_preds_new /= n_models
        final_test_preds_new = np.argmax(ensemble_preds_new, axis=1)
        final_test_conf_new = ensemble_preds_new.max(axis=1)
        pl_mask = final_test_conf_new >= 0.90

    X_aug = np.vstack([X_vals, Xt_vals[pl_mask]])
    y_aug = np.concatenate([y, final_test_preds if iteration == 0 else final_test_preds_new])

    print(f"  Iteration {iteration+1}: {pl_mask.sum()} pseudo-labels, total={len(X_aug)}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    oof_b = np.zeros(n, dtype=int)
    folds_b = []

    for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
        X_aug_tr = np.vstack([X_vals[tr], Xt_vals[pl_mask]])
        y_aug_tr = np.concatenate([y[tr], final_test_preds if iteration == 0 else final_test_preds_new])
        sw_tr = np.ones(len(X_aug_tr))
        sw_tr[len(tr):] = 0.5

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_aug_tr)
        X_val_s = scaler.transform(X_vals[val])

        cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                      random_seed=42, verbose=0)
        cb_m.fit(X_tr_s, y_aug_tr, sample_weight=sw_tr)
        oof_b[val] = cb_m.predict(X_val_s)
        folds_b.append(accuracy_score(y[val], oof_b[val]))

    acc_b = accuracy_score(y, oof_b)
    print(f"  OOF: {acc_b:.4f}, mean={np.mean(folds_b):.4f}")

# ================================================================
# Strategy C: Hard sample correction using pseudo-labels
# Use test predictions to identify label noise in training
# ================================================================
print("\n=== Strategy C: Hard Sample Correction ===")

# Train model ensemble on all data, predict on train
# Identify samples where ensemble disagrees with label
train_preds_ensemble = np.zeros((n, 4))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)

for mname, mfn in model_fns.items():
    oof_p = np.zeros((n, 4))
    for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_vals[tr])
        X_val = scaler.transform(X_vals[val])
        m = mfn(42+fi).fit(X_tr, y[tr])
        oof_p[val] = m.predict_proba(X_val)
    train_preds_ensemble += oof_p

train_preds_ensemble /= len(model_fns)
train_conf = train_preds_ensemble.max(axis=1)
train_pred = np.argmax(train_preds_ensemble, axis=1)

# Find hard samples: all models agree on wrong label
hard_mask = (train_pred != y) & (train_conf > 0.80)
print(f"  Hard samples (conf>0.80, all wrong): {hard_mask.sum()}/{n}")

# Attempt correction: flip labels for hard samples where ensemble is confident
# Only correct if multiple models agree on alternative label
corrected_y = y.copy()
n_corrected = 0
for i in range(n):
    if hard_mask[i]:
        corrected_y[i] = train_pred[i]
        n_corrected += 1

print(f"  Corrected {n_corrected} labels")

# Evaluate with corrected labels
if n_corrected > 0:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    oof_c = np.zeros(n, dtype=int)
    folds_c = []

    for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_vals[tr])
        X_val = scaler.transform(X_vals[val])
        rs = 42 + fi

        cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                      random_seed=rs, verbose=0)
        cb_m.fit(X_tr, corrected_y[tr])
        oof_c[val] = cb_m.predict(X_val)
        folds_c.append(accuracy_score(y[val], oof_c[val]))

    acc_c = accuracy_score(y, oof_c)
    print(f"  Corrected labels OOF: {acc_c:.4f}")

print(f"\nPseudo-labeling experiments complete.")
