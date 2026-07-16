"""
Reproduce baseline ~0.65 accuracy using simple models.
Stage 1: initialization + data audit + baseline reproduction.
"""
import numpy as np
import pandas as pd
import json
import os
import time
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

RANDOM_SEEDS = [42, 123, 2026]
DATA_DIR = Path("data")
EXP_DIR = Path("experiments")
EXP_DIR.mkdir(exist_ok=True)
(EXP_DIR / "oof_predictions").mkdir(exist_ok=True)
(EXP_DIR / "test_predictions").mkdir(exist_ok=True)
(EXP_DIR / "feature_importance").mkdir(exist_ok=True)
(EXP_DIR / "confusion_matrix").mkdir(exist_ok=True)
(EXP_DIR / "models").mkdir(exist_ok=True)
(EXP_DIR / "reports").mkdir(exist_ok=True)


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sample_sub


def data_audit(train, test):
    """Comprehensive data audit."""
    print("=" * 60)
    print("DATA AUDIT")
    print("=" * 60)

    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Train columns: {list(train.columns)}")
    print(f"Test columns: {list(test.columns)}")

    # Target distribution
    print(f"\nTarget distribution:\n{train['target'].value_counts().sort_index()}")

    # Missing values
    print(f"\nMissing values train: {train.isnull().sum().sum()}")
    print(f"Missing values test: {test.isnull().sum().sum()}")

    # Duplicates
    dup_train = train.drop(columns=['target']).duplicated().sum()
    print(f"\nDuplicate rows (excl target): {dup_train}")

    # Constant features
    drop_cols = ['id', 'target']
    train_feat = train.drop(columns=drop_cols)
    test_feat = test.drop(columns=['id'])

    const_cols = [c for c in train_feat.columns if train_feat[c].nunique() <= 1]
    print(f"\nConstant features: {const_cols if const_cols else 'None'}")

    # Near-constant features
    near_const = []
    for c in train_feat.columns:
        vc = train_feat[c].value_counts(normalize=True)
        if vc.iloc[0] > 0.99:
            near_const.append(c)
    print(f"Near-constant features (>99% same value): {near_const if near_const else 'None'}")

    # ID-target correlation
    id_corr = train['id'].corr(train['target'])
    print(f"\nID-target correlation: {id_corr:.6f}")
    id_corr_abs = train['id'].corr(train['target'].astype(float))
    print(f"ID-target (float) correlation: {id_corr_abs:.6f}")

    # Check for sort-order leakage: does id order correlate with target
    id_sorted = train.sort_values('id')
    rolling_target = id_sorted['target'].rolling(100, min_periods=1).mean()
    print(f"Rolling target (window=100, sorted by ID) std: {rolling_target.std():.4f}")
    print(f"Overall target std: {train['target'].std():.4f}")

    # Adversarial validation
    from sklearn.ensemble import RandomForestClassifier
    train_adv = train_feat.copy()
    test_adv = test_feat.copy()
    y_adv = np.concatenate([np.zeros(len(train_adv)), np.ones(len(test_adv))])
    X_adv = pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)
    X_adv = X_adv.fillna(0)

    from sklearn.model_selection import cross_val_score
    adv_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    adv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        X_adv, y_adv, cv=adv_cv, scoring='roc_auc'
    )
    print(f"\nAdversarial validation (train vs test) AUC: {adv_scores.mean():.4f} ± {adv_scores.std():.4f}")
    if adv_scores.mean() > 0.85:
        print("WARNING: High adversarial AUC — train and test are very different!")
    elif adv_scores.mean() > 0.70:
        print("Moderate difference between train and test distributions.")
    else:
        print("Train and test distributions are similar — good.")

    # Feature correlation with target
    corrs = train_feat.corrwith(train['target']).abs().sort_values(ascending=False)
    print(f"\nTop 10 features by absolute correlation with target:")
    for feat, corr_val in corrs.head(10).items():
        print(f"  {feat}: {corr_val:.4f}")

    return {
        'shape_train': train.shape,
        'shape_test': test.shape,
        'missing_train': train.isnull().sum().sum(),
        'missing_test': test.isnull().sum().sum(),
        'duplicates': dup_train,
        'constant_features': const_cols,
        'near_constant_features': near_const,
        'id_target_corr': id_corr,
        'adv_auc_mean': adv_scores.mean(),
        'adv_auc_std': adv_scores.std(),
        'target_distribution': train['target'].value_counts().to_dict()
    }


def get_features(train, test):
    """Extract features, dropping id and target."""
    drop_cols = ['id']
    if 'target' in train.columns:
        y = train['target'].values
        X = train.drop(columns=drop_cols + ['target'])
    else:
        y = None
        X = train.drop(columns=drop_cols)

    X_test = test.drop(columns=['id'])

    # Ensure same columns
    assert list(X.columns) == list(X_test.columns), "Column mismatch!"
    return X.values, y, X_test.values, X.columns.tolist()


def evaluate_model_cv(X, y, model, model_name, seed=42, n_repeats=1):
    """Evaluate model with cross-validation."""
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=n_repeats, random_state=seed
    )

    fold_metrics = []
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros((len(y), 4))
    oof_fold = np.zeros(len(y))

    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_f, X_val_f = X[train_idx], X[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_f)
        X_val_scaled = scaler.transform(X_val_f)

        # Train
        model_clone = model
        model_clone.fit(X_train_scaled, y_train_f)

        # Predict
        y_val_pred = model_clone.predict(X_val_scaled)
        y_val_prob = model_clone.predict_proba(X_val_scaled)
        y_train_pred = model_clone.predict(X_train_scaled)

        # Metrics
        val_acc = accuracy_score(y_val_f, y_val_pred)
        val_f1 = f1_score(y_val_f, y_val_pred, average='macro')
        val_balanced = balanced_accuracy_score(y_val_f, y_val_pred)
        train_acc = accuracy_score(y_train_f, y_train_pred)

        fold_metrics.append({
            'fold': fold_idx,
            'accuracy': val_acc,
            'macro_f1': val_f1,
            'balanced_accuracy': val_balanced,
            'train_accuracy': train_acc,
            'overfit_gap': train_acc - val_acc,
        })

        oof_preds[val_idx] = y_val_pred
        oof_probs[val_idx] = y_val_prob
        oof_fold[val_idx] = fold_idx

    elapsed = time.time() - start_time

    # Aggregate
    accs = [m['accuracy'] for m in fold_metrics]
    f1s = [m['macro_f1'] for m in fold_metrics]
    bas = [m['balanced_accuracy'] for m in fold_metrics]

    results = {
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'min_fold': float(np.min(accs)),
        'max_fold': float(np.max(accs)),
        'median_accuracy': float(np.median(accs)),
        'macro_f1': float(np.mean(f1s)),
        'balanced_accuracy': float(np.mean(bas)),
        'train_score': float(np.mean([m['train_accuracy'] for m in fold_metrics])),
        'overfit_gap': float(np.mean([m['overfit_gap'] for m in fold_metrics])),
        'fold_details': fold_metrics,
        'runtime': elapsed,
        'oof_predictions': oof_preds,
        'oof_probabilities': oof_probs,
        'oof_fold': oof_fold,
    }

    print(f"\n{model_name} — CV results (seed={seed}):")
    print(f"  Mean accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"  Min fold: {results['min_fold']:.4f}, Max fold: {results['max_fold']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Balanced acc: {results['balanced_accuracy']:.4f}")
    print(f"  Train-val gap: {results['overfit_gap']:.4f}")
    print(f"  Runtime: {elapsed:.1f}s")

    return results


def log_experiment(exp_id, parent, hypothesis, feature_set, model_name, params,
                   cv_strategy, seed, metrics, accepted, notes=""):
    """Log experiment to CSV."""
    log_path = EXP_DIR / "experiment_log.csv"
    row = {
        'experiment_id': exp_id,
        'parent_experiment': parent,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hypothesis': hypothesis,
        'feature_set': feature_set,
        'model': model_name,
        'parameters': str(params),
        'cv_strategy': cv_strategy,
        'seed': seed,
        'mean_accuracy': f"{metrics['mean_accuracy']:.6f}",
        'std_accuracy': f"{metrics['std_accuracy']:.6f}",
        'minimum_fold': f"{metrics['min_fold']:.6f}",
        'macro_f1': f"{metrics['macro_f1']:.6f}",
        'balanced_accuracy': f"{metrics['balanced_accuracy']:.6f}",
        'train_score': f"{metrics['train_score']:.6f}",
        'overfit_gap': f"{metrics['overfit_gap']:.6f}",
        'runtime': f"{metrics['runtime']:.2f}",
        'accepted': accepted,
        'notes': notes,
    }
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)
    return exp_id


def save_best_config(metrics, model_name, params, features, exp_id):
    """Save best model configuration."""
    config = {
        'experiment_id': exp_id,
        'model': model_name,
        'parameters': params,
        'features': features,
        'metrics': {k: v for k, v in metrics.items()
                    if k not in ['oof_predictions', 'oof_probabilities', 'oof_fold', 'fold_details']},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(EXP_DIR / "best_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved best config (exp_id={exp_id})")


def reproduce_baseline(X, y):
    """Reproduce ~0.65 baseline with various simple models."""
    print("\n" + "=" * 60)
    print("REPRODUCING BASELINE ~0.65")
    print("=" * 60)

    SEED = 42
    N_REPEATS = 2
    best_acc = 0.0
    best_info = None

    # Strategy 1: SVC RBF
    print("\n--- SVC RBF ---")
    svc = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED)
    res = evaluate_model_cv(X, y, svc, "SVC RBF", seed=SEED, n_repeats=N_REPEATS)
    log_experiment("EXP-001", "None", "SVC RBF baseline", "raw features", "SVC RBF",
                   "{C:10, gamma:scale}", "RepeatedStratifiedKFold(5,2)", SEED,
                   res, "Ya" if res['mean_accuracy'] > best_acc else "Tidak",
                   "Baseline SVC RBF")
    if res['mean_accuracy'] > best_acc:
        best_acc = res['mean_accuracy']
        best_info = ('SVC RBF', res, "{C:10, gamma:scale}")

    # Strategy 2: RandomForest
    print("\n--- RandomForest ---")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4,
                                random_state=SEED, n_jobs=-1)
    res = evaluate_model_cv(X, y, rf, "RandomForest", seed=SEED, n_repeats=N_REPEATS)
    log_experiment("EXP-002", "None", "RandomForest baseline", "raw features", "RandomForest",
                   "{n_estimators:300, max_depth:12, min_samples_leaf:4}",
                   "RepeatedStratifiedKFold(5,2)", SEED,
                   res, "Ya" if res['mean_accuracy'] > best_acc else "Tidak",
                   "Baseline RF")
    if res['mean_accuracy'] > best_acc:
        best_acc = res['mean_accuracy']
        best_info = ('RandomForest', res, "{n_estimators:300, max_depth:12}")

    # Strategy 3: Logistic Regression
    print("\n--- LogisticRegression ---")
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                            random_state=SEED, n_jobs=-1)
    res = evaluate_model_cv(X, y, lr, "LogisticRegression", seed=SEED, n_repeats=N_REPEATS)
    log_experiment("EXP-003", "None", "Logistic Regression baseline", "raw features", "LogisticRegression",
                   "{C:1.0, solver:lbfgs, multi_class:multinomial}",
                   "RepeatedStratifiedKFold(5,2)", SEED,
                   res, "Ya" if res['mean_accuracy'] > best_acc else "Tidak",
                   "Baseline LR")
    if res['mean_accuracy'] > best_acc:
        best_acc = res['mean_accuracy']
        best_info = ('LogisticRegression', res, "{C:1.0}")

    # Strategy 4: GradientBoosting
    print("\n--- GradientBoosting ---")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                    min_samples_leaf=5, random_state=SEED)
    res = evaluate_model_cv(X, y, gb, "GradientBoosting", seed=SEED, n_repeats=N_REPEATS)
    log_experiment("EXP-004", "None", "GradientBoosting baseline", "raw features", "GradientBoosting",
                   "{n_estimators:200, max_depth:5, lr:0.1}",
                   "RepeatedStratifiedKFold(5,2)", SEED,
                   res, "Ya" if res['mean_accuracy'] > best_acc else "Tidak",
                   "Baseline GB")
    if res['mean_accuracy'] > best_acc:
        best_acc = res['mean_accuracy']
        best_info = ('GradientBoosting', res, "{n_estimators:200, max_depth:5}")

    # Strategy 5: KNN
    print("\n--- KNN ---")
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance', p=2)
    res = evaluate_model_cv(X, y, knn, "KNN", seed=SEED, n_repeats=N_REPEATS)
    log_experiment("EXP-005", "None", "KNN baseline", "raw features", "KNN",
                   "{n_neighbors:15, weights:distance}",
                   "RepeatedStratifiedKFold(5,2)", SEED,
                   res, "Ya jika lebih baik", "Baseline KNN")

    print("\n" + "=" * 60)
    print(f"BEST BASELINE: {best_info[0]} — Accuracy: {best_acc:.4f}")
    print("=" * 60)

    return best_info


def main():
    print("=" * 60)
    print("DATATHON — AUTOMATED EXPERIMENT PIPELINE")
    print("=" * 60)

    # 1. Load
    train, test, sample_sub = load_data()
    print(f"\nLoaded: train {train.shape}, test {test.shape}, submission {sample_sub.shape}")

    # 2. Audit
    audit = data_audit(train, test)

    # Save audit
    with open(EXP_DIR / "reports" / "data_audit.json", 'w') as f:
        # Convert numpy types
        clean_audit = {}
        for k, v in audit.items():
            if isinstance(v, np.integer):
                clean_audit[k] = int(v)
            elif isinstance(v, np.floating):
                clean_audit[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_audit[k] = v.tolist()
            else:
                clean_audit[k] = v
        json.dump(clean_audit, f, indent=2, default=str)

    # 3. Get features
    X, y, X_test, feature_names = get_features(train, test)
    print(f"\nFeatures: {X.shape[1]}, Train samples: {X.shape[0]}, Test samples: {X_test.shape[0]}")

    # 4. Reproduce baseline
    best_name, best_metrics, best_params = reproduce_baseline(X, y)

    # 5. Save best config
    save_best_config(best_metrics, best_name, best_params, list(feature_names), "EXP-best")

    # 6. OOF predictions dari model terbaik
    if best_name == 'SVC RBF':
        final_model = SVC(C=10, gamma='scale', probability=True, random_state=42)
    elif best_name == 'RandomForest':
        final_model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4,
                                              random_state=42, n_jobs=-1)
    elif best_name == 'LogisticRegression':
        final_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                          multi_class='multinomial', random_state=42, n_jobs=-1)
    elif best_name == 'GradientBoosting':
        final_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                  min_samples_leaf=5, random_state=42)
    else:
        final_model = SVC(C=10, gamma='scale', probability=True, random_state=42)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros((len(y), 4))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_f, X_val_f = X[train_idx], X[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_f)
        X_val_scaled = scaler.transform(X_val_f)

        final_model.fit(X_train_scaled, y_train_f)
        oof_preds[val_idx] = final_model.predict(X_val_scaled)
        oof_probs[val_idx] = final_model.predict_proba(X_val_scaled)

    oof_acc = accuracy_score(y, oof_preds)
    oof_f1 = f1_score(y, oof_preds, average='macro')
    print(f"\nFinal OOF: accuracy={oof_acc:.4f}, macro_f1={oof_f1:.4f}")

    # Save OOF
    np.save(EXP_DIR / "oof_predictions" / "baseline_oof_preds.npy", oof_preds)
    np.save(EXP_DIR / "oof_predictions" / "baseline_oof_probs.npy", oof_probs)
    pd.DataFrame({'id': train['id'], 'target': oof_preds.astype(int)}).to_csv(
        EXP_DIR / "oof_predictions" / "baseline_oof.csv", index=False)

    # Generate test predictions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    final_model.fit(X_scaled, y)
    test_preds = final_model.predict(X_test_scaled)
    test_probs = final_model.predict_proba(X_test_scaled)

    # Save test predictions
    np.save(EXP_DIR / "test_predictions" / "baseline_test_preds.npy", test_preds)
    np.save(EXP_DIR / "test_predictions" / "baseline_test_probs.npy", test_probs)

    # Create submission
    sub = sample_sub.copy()
    sub['target'] = test_preds.astype(int)
    sub.to_csv(EXP_DIR / "submission_baseline.csv", index=False)
    print(f"\nBaseline submission saved.")

    print("\n" + "=" * 60)
    print("BASELINE REPRODUCTION COMPLETE")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"Best accuracy: {best_metrics['mean_accuracy']:.4f}")
    print(f"This is your starting benchmark. Next: experiment loop.")


if __name__ == "__main__":
    main()
