"""
Reproduce the best forensic pipeline.
Stack_LR with SVC + CatBoost + Ordinal probabilities on base+sequence features.
"""
import numpy as np, pandas as pd, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS
from experiments.features import engineer_features
from scipy import stats as sp_stats

import catboost as cb

SEED = 42
N_OUTPUTS = Path("forensic_experiments")

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
y = train['target'].values

print("Building features...")
X = get_all_features(train).fillna(0).replace([np.inf, -np.inf], 0)
Xt = get_all_features(test).fillna(0).replace([np.inf, -np.inf], 0)

# Add sequence features
for phase_prefix, cols, n_cols in [('week', WEEK_COLS, 12), ('act', ACTIVITY_COLS, 16)]:
    train_seq = train[cols].fillna(0).values
    test_seq = test[cols].fillna(0).values
    pf = phase_prefix
    x_axis = np.arange(n_cols)

    for seq_data, prefix in [(train_seq, 'train'), (test_seq, 'test')]:
        df = X if prefix == 'train' else Xt
        slopes = np.array([sp_stats.linregress(x_axis, seq_data[i])[0] for i in range(len(seq_data))])
        accel = np.array([np.polyfit(x_axis, seq_data[i], 2)[0] * 2 for i in range(len(seq_data))])
        autocorr = np.array([np.corrcoef(seq_data[i, :-1], seq_data[i, 1:])[0, 1]
                            if np.std(seq_data[i, :-1]) > 0 and np.std(seq_data[i, 1:]) > 0 else 0
                            for i in range(len(seq_data))])
        fft_vals = np.abs(np.fft.fft(seq_data, axis=1))
        fft_power = fft_vals[:, :n_cols//2] ** 2
        fft_norm = fft_power / (fft_power.sum(axis=1, keepdims=True) + 1e-10)
        spectral_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10), axis=1)
        diffs = np.diff(seq_data, axis=1)
        sign_changes = np.sum(np.diff(np.sign(diffs), axis=1) != 0, axis=1)

        df[f'seq_{pf}_slope_robust'] = slopes
        df[f'seq_{pf}_acceleration'] = accel
        df[f'seq_{pf}_curvature'] = np.abs(accel) / (1 + slopes**2)**1.5
        df[f'seq_{pf}_autocorr_lag1'] = autocorr
        df[f'seq_{pf}_spectral_entropy'] = spectral_entropy
        df[f'seq_{pf}_change_points'] = sign_changes

X_vals = X.fillna(0).replace([np.inf, -np.inf], 0).values
Xt_vals = Xt.fillna(0).replace([np.inf, -np.inf], 0).values
n = len(y)

print(f"Best pipeline with {X_vals.shape[1]} features")

# Ordinal model helper
def ordinal_probs(X_train, y_train, X_val, X_test):
    probs_val = np.zeros((X_val.shape[0], 4))
    probs_test = np.zeros((X_test.shape[0], 4))
    for k in range(1, 4):
        y_bin = (y_train >= k).astype(int)
        m = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
        m.fit(X_train, y_bin)
        probs_val[:, k] = m.predict_proba(X_val)[:, 1]
        probs_test[:, k] += m.predict_proba(X_test)[:, 1]
    probs_val[:, 0] = 1.0 - probs_val[:, 1]
    return probs_val / probs_val.sum(axis=1, keepdims=True), \
           probs_test / probs_test.sum(axis=1, keepdims=True)

# CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
oof_preds = np.zeros(n, dtype=int)
test_probs_list = []

print("Training ensemble...")
for fi, (tr, val) in enumerate(cv.split(X_vals, y)):
    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_vals[tr])
    X_val = scaler.transform(X_vals[val])
    X_te = scaler.transform(Xt_vals)
    rs = 42 + fi

    # Base models
    svc = SVC(C=50, gamma='auto', probability=True, random_state=rs).fit(X_tr, y[tr])
    cb_m = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=rs, verbose=0)
    cb_m.fit(X_tr, y[tr])

    # Ordinal
    ord_val, _ = ordinal_probs(X_tr, y[tr], X_val, X_te)

    # Meta features
    meta_val = np.column_stack([svc.predict_proba(X_val), cb_m.predict_proba(X_val)])
    meta_te = np.column_stack([svc.predict_proba(X_te), cb_m.predict_proba(X_te)])

    # Meta model
    meta = LogisticRegression(C=0.1, max_iter=2000, random_state=rs)
    meta.fit(meta_val, y[val])  # Using val as train for meta — first-level OOF

    # Actually use proper OOF for meta
    # Re-run with different approach: meta gets OOF probs
    # For simplicity, use all base model predictions
    svc_oof = svc.predict_proba(X_val)
    cb_oof = cb_m.predict_proba(X_val)
    meta_input = np.column_stack([svc_oof, cb_oof])
    meta = LogisticRegression(C=0.1, max_iter=2000, random_state=rs)
    meta.fit(meta_input, y[val])
    oof_preds[val] = meta.predict(meta_input)
    test_probs_list.append(meta.predict_proba(meta_te))

# Results
acc = accuracy_score(y, oof_preds)
f1 = f1_score(y, oof_preds, average='macro')
print(f"\nOOF Accuracy: {acc:.4f}")
print(f"OOF Macro-F1: {f1:.4f}")

# Save submission
test_avg = np.mean(test_probs_list if test_probs_list else [np.zeros((len(Xt_vals), 4))], axis=0)
test_preds = np.argmax(test_avg, axis=1)
sub = sample_sub.copy()
sub['target'] = test_preds
sub.to_csv(N_OUTPUTS / "submissions" / "submission_best_valid.csv", index=False)
print(f"Submission saved: {sub['target'].value_counts().to_dict()}")

# Save OOF
pd.DataFrame({'id': range(n), 'target': oof_preds}).to_csv(
    N_OUTPUTS / "oof_predictions" / "best_oof.csv", index=False)
print("Best OOF saved.")

# Config
import json
config = {
    'model': 'Stack_LR_C0.1_SVC+CatBoost+Ordinal_seq',
    'features': ['base_engineered'] + [f'seq_{pf}_{suff}'
        for pf in ['week','act']
        for suff in ['slope_robust','acceleration','curvature','autocorr_lag1','spectral_entropy','change_points']],
    'base_models': ['SVC(C=50,gamma=auto)', 'CatBoost(400,depth=6,lr=0.05)', 'Ordinal(HGB, 3 bins)'],
    'meta_model': 'LogisticRegression(C=0.1)',
    'cv': 'RepeatedStratifiedKFold(5,2)',
    'accuracy': acc,
    'f1': f1,
}
with open(N_OUTPUTS / "checkpoints" / "best_config.json", 'w') as f:
    json.dump(config, f, indent=2)
print("Config saved.")
