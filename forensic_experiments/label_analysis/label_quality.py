"""
Phase 4: Label Quality Analysis.
Identify hard samples, possible mislabels, model disagreement patterns.
"""
import numpy as np, pandas as pd, json, sys, warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from scipy.special import softmax

from experiments.features import get_all_features
from forensic_experiments.core import load_data, FOR_DIR, get_cv, evaluate_cv

import catboost as cb

train, test, sample = load_data()
y = train['target'].values
X_all = get_all_features(train).fillna(0).replace([np.inf, -np.inf], 0)
X_vals = X_all.values

print(f"Data: {X_vals.shape}")

# ================================================================
# 1. MULTI-MODEL OOF PREDICTIONS
# ================================================================
print("\n" + "="*60)
print("GENERATING MULTI-MODEL OOF PREDICTIONS")
print("="*60)

cv = get_cv(42)
n = len(y)
n_seeds = 3
n_folds = 10

models_dict = {
    'SVC': lambda s: SVC(C=50, gamma='auto', probability=True, random_state=s),
    'CatBoost': lambda s: cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, random_seed=s, verbose=0),
    'RF': lambda s: RandomForestClassifier(n_estimators=400, max_depth=14, min_samples_leaf=3, random_state=s, n_jobs=-1),
    'ET': lambda s: ExtraTreesClassifier(n_estimators=400, max_depth=10, min_samples_leaf=3, random_state=s, n_jobs=-1),
    'HGB': lambda s: HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=s),
}

# Collect predictions from multiple models and seeds
all_preds = {}  # (model, seed) -> oof predictions
all_confs = {}  # (model, seed) -> max probability

for mname, mfn in models_dict.items():
    for seed in [42, 123, 2026]:
        cv_s = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
        oof_p = np.zeros((n, 4))
        oof_pred = np.zeros(n, dtype=int)
        for tr, val in cv_s.split(X_vals, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_vals[tr])
            X_val = scaler.transform(X_vals[val])
            m = mfn(seed)
            m.fit(X_tr, y[tr])
            if hasattr(m, 'predict_proba'):
                prob = m.predict_proba(X_val)
                oof_p[val] = prob
                oof_pred[val] = np.argmax(prob, axis=1)
            else:
                oof_pred[val] = m.predict(X_val)

        key = f"{mname}_s{seed}"
        all_preds[key] = oof_pred
        all_confs[key] = np.max(oof_p, axis=1) if oof_p.sum() > 0 else np.zeros(n)
        acc = accuracy_score(y, oof_pred)
        print(f"  {key:20s} acc={acc:.4f}")

# ================================================================
# 2. CONSENSUS ANALYSIS
# ================================================================
print("\n" + "="*60)
print("SAMPLE-WISE CONSENSUS ANALYSIS")
print("="*60)

n_models = len(all_preds)
model_names = list(all_preds.keys())

# For each sample: how many models are correct?
correct_count = np.zeros(n, dtype=int)
for key in model_names:
    correct_count += (all_preds[key] == y).astype(int)

# For each sample: how many seeds predict the same dominant class?
dominant_prediction = np.zeros(n, dtype=int)
majority_count = np.zeros(n, dtype=int)
for i in range(n):
    preds_this_sample = [all_preds[k][i] for k in model_names]
    from collections import Counter
    counter = Counter(preds_this_sample)
    dominant_prediction[i] = counter.most_common(1)[0][0]
    majority_count[i] = counter.most_common(1)[0][1]

# Confidence score: fraction of models that agree on the dominant prediction
confidence = majority_count / n_models

# Model agreement: fraction of models that agree on any prediction
# Already computed as majority_count/n_models = confidence

# ================================================================
# 3. BUILD SAMPLE TABLE
# ================================================================
print("\nBuilding sample analysis table...")

sample_data = pd.DataFrame({
    'id': np.arange(n),
    'target': y,
    'dominant_prediction': dominant_prediction,
    'correct_models': correct_count,
    'total_models': n_models,
    'correct_ratio': correct_count / n_models,
    'majority_count': majority_count,
    'confidence': confidence,
    'majority_correct': (dominant_prediction == y).astype(int),
})

# Add per-model predictions
for key in model_names:
    sample_data[f'pred_{key}'] = all_preds[key]
    sample_data[f'correct_{key}'] = (all_preds[key] == y).astype(int)

# Classification
def classify_row(row):
    if row['correct_ratio'] >= 0.8:
        if row['majority_correct']:
            return 'EASY_CORRECT'
        else:
            return 'HIGH_CONFIDENCE_WRONG'
    elif row['correct_ratio'] >= 0.5:
        if row['majority_correct']:
            return 'BORDERLINE_CORRECT'
        else:
            return 'BORDERLINE_WRONG'
    elif row['correct_ratio'] >= 0.2:
        return 'UNCERTAIN'
    else:
        return 'CONSISTENTLY_WRONG'

sample_data['category'] = sample_data.apply(classify_row, axis=1)

# ================================================================
# 4. IDENTIFY POSSIBLE MISLABELS
# ================================================================
print("Identifying possible mislabels...")

# Criteria:
#   - Majority/confidence >= 0.80 predicts a class different from actual
#   - Mean confidence >= 0.80 (models are confident but wrong)
#   - At least 80% of models agree
possible_mislabels = sample_data[
    (sample_data['confidence'] >= 0.80) &
    (sample_data['dominant_prediction'] != sample_data['target'])
].copy()

possible_mislabels['mean_confidence'] = possible_mislabels['confidence']

print(f"  Possible mislabels: {len(possible_mislabels)}")
if len(possible_mislabels) > 0:
    print("\n  Top possible mislabels:")
    for _, row in possible_mislabels.sort_values('confidence', ascending=False).head(20).iterrows():
        print(f"    ID={row['id']:4d} actual={int(row['target'])} "
              f"pred={int(row['dominant_prediction'])} "
              f"confidence={row['confidence']:.2f} "
              f"correct_models={int(row['correct_models'])}/{n_models}")

# Also check hard samples (all models disagree or low confidence)
hard_samples = sample_data[
    (sample_data['correct_ratio'] < 0.5) |
    (sample_data['confidence'] < 0.5)
].copy()

print(f"\n  Hard samples (low consensus): {len(hard_samples)}")

# ================================================================
# 5. CATEGORY DISTRIBUTION
# ================================================================
print("\nCategory distribution:")
cat_dist = sample_data['category'].value_counts()
for cat, count in cat_dist.items():
    pct = count / n * 100
    print(f"  {cat:25s}: {count:4d} ({pct:.1f}%)")

# Cross-tab: actual vs dominant prediction
xtab = pd.crosstab(sample_data['target'], sample_data['dominant_prediction'],
                     rownames=['Actual'], colnames=['Dominant_Prediction'])
print(f"\nConfusion matrix (majority vote):")
print(xtab)

# ================================================================
# 6. SAVE
# ================================================================
sample_data.to_csv(FOR_DIR / "label_analysis" / "hard_samples.csv", index=False)
possible_mislabels.to_csv(FOR_DIR / "label_analysis" / "possible_mislabels.csv", index=False)

# Summary report
report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_samples': n,
    'n_models': n_models,
    'models_used': list(models_dict.keys()),
    'seeds_used': [42, 123, 2026],
    'category_distribution': cat_dist.to_dict(),
    'possible_mislabels': len(possible_mislabels),
    'hard_samples': len(hard_samples),
    'majority_vote_accuracy': float(accuracy_score(y, dominant_prediction)),
    'majority_vote_f1': float(f1_score(y, dominant_prediction, average='macro')),
    'easy_correct': int(np.sum(sample_data['category'] == 'EASY_CORRECT')),
    'high_confidence_wrong': int(np.sum(sample_data['category'] == 'HIGH_CONFIDENCE_WRONG')),
    'consistently_wrong': int(np.sum(sample_data['category'] == 'CONSISTENTLY_WRONG')),
    'uncertain': int(np.sum(sample_data['category'] == 'UNCERTAIN')),
}

with open(FOR_DIR / "label_analysis" / "label_quality_summary.json", 'w') as f:
    json.dump(report, f, indent=2)

# Also save model disagreement matrix
disagreement_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
for k1 in model_names:
    for k2 in model_names:
        disagreement_matrix.loc[k1, k2] = np.mean(all_preds[k1] != all_preds[k2])

disagreement_matrix.to_csv(FOR_DIR / "model_analysis" / "model_disagreement.csv")
print(f"\nModel disagreement matrix:")
print(disagreement_matrix.round(4))

print(f"\n{'='*60}")
print("LABEL QUALITY ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Majority vote accuracy: {report['majority_vote_accuracy']:.4f}")
print(f"Possible mislabels: {len(possible_mislabels)}")
print(f"High-confidence wrong: {report['high_confidence_wrong']} samples")
print(f"Consistently wrong: {report['consistently_wrong']} samples")
print(f"Uncertain: {report['uncertain']} samples")
print(f"Easy correct: {report['easy_correct']} samples")
