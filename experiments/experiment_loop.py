"""
Automated Experiment Loop for Datathon.
Runs experiments continuously until target 0.70 is reached or plateau conditions met.
"""
import numpy as np
import pandas as pd
import json
import os
import time
import sys
import warnings
import traceback
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, log_loss)

from experiments.features import get_all_features, WEEK_COLS, ACTIVITY_COLS

SEED = 42
DATA_DIR = Path("data")
EXP_DIR = Path("experiments")
EXP_DIR.mkdir(exist_ok=True)

# Model configurations
MODELS = {
    'SVC_RBF': lambda rs: SVC(C=10, gamma='scale', probability=True, random_state=rs),
    'SVC_RBF_strong': lambda rs: SVC(C=50, gamma='auto', probability=True, random_state=rs),
    'RandomForest': lambda rs: RandomForestClassifier(n_estimators=400, max_depth=14,
                                                       min_samples_leaf=3, random_state=rs, n_jobs=-1),
    'ExtraTrees': lambda rs: ExtraTreesClassifier(n_estimators=400, max_depth=14,
                                                   min_samples_leaf=3, random_state=rs, n_jobs=-1),
    'GradientBoosting': lambda rs: GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                                                learning_rate=0.05, min_samples_leaf=5,
                                                                random_state=rs),
    'HistGB': lambda rs: HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                                         learning_rate=0.05, random_state=rs),
    'LogisticRegression': lambda rs: LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000,
                                                          random_state=rs, n_jobs=-1),
    'Ridge': lambda rs: RidgeClassifier(alpha=1.0, random_state=rs),
    'KNN': lambda rs: KNeighborsClassifier(n_neighbors=15, weights='distance'),
    'LDA': lambda rs: LinearDiscriminantAnalysis(),
}


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sample_sub


def load_experiment_log():
    log_path = EXP_DIR / "experiment_log.csv"
    if log_path.exists():
        return pd.read_csv(log_path)
    return pd.DataFrame()


def evaluate_model_cv(X, y, model, model_name, X_test=None, seed=42, n_repeats=2):
    """Evaluate with RepeatedStratifiedKFold CV. Returns metrics + OOF."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    n_total = len(y)

    oof_preds = np.zeros(n_total)
    oof_probs = np.zeros((n_total, 4))
    oof_fold = np.zeros(n_total)
    fold_metrics = []
    test_preds_list = []
    test_probs_list = []

    start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Model clone
        try:
            m = model.__class__(**model.get_params())
            if hasattr(m, 'random_state'):
                m.set_params(random_state=seed + fold_idx)
        except Exception:
            m = model

        m.fit(X_tr_s, y_tr)

        y_val_pred = m.predict(X_val_s)
        y_val_prob = m.predict_proba(X_val_s) if hasattr(m, 'predict_proba') else None
        y_tr_pred = m.predict(X_tr_s)

        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        val_bal = balanced_accuracy_score(y_val, y_val_pred)
        train_acc = accuracy_score(y_tr, y_tr_pred)

        fold_metrics.append({
            'fold': fold_idx,
            'accuracy': val_acc,
            'macro_f1': val_f1,
            'balanced_accuracy': val_bal,
            'train_accuracy': train_acc,
            'overfit_gap': train_acc - val_acc,
        })

        oof_preds[val_idx] = y_val_pred
        if y_val_prob is not None:
            oof_probs[val_idx] = y_val_prob
        oof_fold[val_idx] = fold_idx

        # Test predictions
        if X_test is not None:
            X_test_s = scaler.transform(X_test)
            test_preds_list.append(m.predict(X_test_s))
            if hasattr(m, 'predict_proba'):
                test_probs_list.append(m.predict_proba(X_test_s))

    elapsed = time.time() - start

    accs = [m['accuracy'] for m in fold_metrics]
    f1s = [m['macro_f1'] for m in fold_metrics]

    oof_acc = accuracy_score(y, oof_preds)
    oof_f1 = f1_score(y, oof_preds, average='macro')

    results = {
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'min_fold': float(np.min(accs)),
        'max_fold': float(np.max(accs)),
        'median_accuracy': float(np.median(accs)),
        'macro_f1': float(oof_f1),
        'balanced_accuracy': float(np.mean([m['balanced_accuracy'] for m in fold_metrics])),
        'train_score': float(np.mean([m['train_accuracy'] for m in fold_metrics])),
        'overfit_gap': float(np.mean([m['overfit_gap'] for m in fold_metrics])),
        'fold_details': fold_metrics,
        'runtime': elapsed,
        'oof_predictions': oof_preds,
        'oof_probabilities': oof_probs,
        'oof_fold': oof_fold,
        'oof_accuracy': float(oof_acc),
    }

    if test_preds_list:
        results['test_preds'] = np.array(test_preds_list)
        if test_probs_list:
            results['test_probs'] = np.array(test_probs_list)

    print(f"\n{model_name} — OOF: {oof_acc:.4f} (±{results['std_accuracy']:.4f}), "
          f"F1={oof_f1:.4f}, min_fold={results['min_fold']:.4f}, gap={results['overfit_gap']:.3f}")

    return results


def log_experiment(exp_id, parent, hypothesis, feature_set, model_name, params,
                   cv_strategy, seed, metrics, accepted, notes=""):
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
        'mean_accuracy': f"{metrics.get('oof_accuracy', metrics.get('mean_accuracy', 0)):.6f}",
        'std_accuracy': f"{metrics['std_accuracy']:.6f}",
        'minimum_fold': f"{metrics['min_fold']:.6f}",
        'macro_f1': f"{metrics['macro_f1']:.6f}",
        'balanced_accuracy': f"{metrics['balanced_accuracy']:.6f}",
        'train_score': f"{metrics['train_score']:.6f}",
        'overfit_gap': f"{metrics['overfit_gap']:.6f}",
        'runtime': f"{metrics['runtime']:.2f}",
        'accepted': accepted,
        'notes': notes[:200] if notes else "",
    }
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)
    return exp_id


def save_checkpoint(exp_id, model_name, metrics, feature_set, params):
    """Save best checkpoint."""
    config = {
        'experiment_id': exp_id,
        'model': model_name,
        'feature_set': feature_set,
        'parameters': params,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'mean_accuracy': metrics['mean_accuracy'],
            'std_accuracy': metrics['std_accuracy'],
            'min_fold': metrics['min_fold'],
            'macro_f1': metrics['macro_f1'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'overfit_gap': metrics['overfit_gap'],
        }
    }
    with open(EXP_DIR / "best_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Save OOF
    np.save(EXP_DIR / "oof_predictions" / "best_oof_preds.npy", metrics['oof_predictions'])
    np.save(EXP_DIR / "oof_predictions" / "best_oof_probs.npy", metrics['oof_probabilities'])
    np.save(EXP_DIR / "oof_predictions" / "best_oof_fold.npy", metrics['oof_fold'])
    pd.DataFrame({'id': range(len(metrics['oof_predictions'])),
                   'target': metrics['oof_predictions'].astype(int)}).to_csv(
        EXP_DIR / "oof_predictions" / "best_oof.csv", index=False)

    print(f"  >>> Checkpoint saved (exp={exp_id}, acc={metrics.get('oof_accuracy', metrics['mean_accuracy']):.4f})")


def make_submission(test_preds, sample_sub, filename):
    """Save submission file."""
    sub = sample_sub.copy()
    sub['target'] = test_preds.astype(int)
    sub.to_csv(EXP_DIR / filename, index=False)
    print(f"  Submission saved: {filename}")


def run_experiment(exp_id, parent, hypothesis, feature_set_name, model_name,
                   model_fn, X_train, y_train, X_test, sample_sub, seed=42,
                   n_repeats=2):
    """Run single experiment end-to-end."""
    print(f"\n{'='*60}")
    print(f"EXP-{exp_id}: {model_name} on {feature_set_name}")
    print(f"  Hypothesis: {hypothesis}")
    print(f"{'='*60}")

    model = model_fn(seed)
    params = str(model.get_params())

    try:
        results = evaluate_model_cv(X_train, y_train, model, model_name,
                                     X_test=X_test, seed=seed, n_repeats=n_repeats)

        accepted = "Ya" if results['oof_accuracy'] >= best_score.get('accuracy', 0) + 0.002 else "Tidak"

        # Save test predictions if available
        if 'test_preds' in results:
            # Majority vote across folds
            test_preds_mode = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), 0, results['test_preds'])
            make_submission(test_preds_mode, sample_sub, f"submission_{exp_id}.csv")

        log_experiment(exp_id, parent, hypothesis, feature_set_name, model_name,
                       params, f"RepeatedStratifiedKFold(5,{n_repeats})", seed,
                       results, accepted, "")

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return None


def print_status(best, log_df, experiment_count, no_improvement_count):
    """Print current experiment status."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT STATUS — Total: {experiment_count}")
    print(f"{'='*60}")
    print(f"Best OOF accuracy: {best.get('accuracy', 0):.4f} (model={best.get('model', '-')})")
    print(f"Best macro F1:    {best.get('macro_f1', 0):.4f}")
    print(f"No-improvement count: {no_improvement_count}")
    if log_df is not None and len(log_df) > 0:
        top = log_df.sort_values('mean_accuracy', ascending=False).head(5)
        print(f"\nTop 5 so far:")
        for _, r in top.iterrows():
            print(f"  {r['experiment_id']}: {r['model']} on {r['feature_set']} = {float(r['mean_accuracy']):.4f}")
    print(f"{'='*60}")
    return best


def main():
    global best_score
    best_score = {'accuracy': 0.0, 'macro_f1': 0.0, 'model': 'None'}

    # ================================================================
    # INIT
    # ================================================================
    train, test, sample_sub = load_data()
    print(f"Loaded: train {train.shape}, test {test.shape}")

    y = train['target'].values
    feature_sets = {
        'raw': lambda df: df.drop(columns=['id', 'target'] if 'target' in df.columns else ['id']),
    }
    print(f"Baseline target distribution: {np.bincount(y)}")

    # ================================================================
    # EXPERIMENT LOOP
    # ================================================================
    experiment_count = 0
    no_improvement_count = 0
    strategy_stage = 1
    exp_id_counter = [5]  # Start after baseline EXPs 1-5

    # Phase 1: Reproduce baseline with feature engineering
    print("\n" + "=" * 60)
    print("PHASE 1: REPRODUCE BASELINE WITH ENGINEERED FEATURES")
    print("=" * 60)

    # EXP-006: SVC RBF + engineered features
    exp_id_counter[0] += 1
    eid = f"EXP-{exp_id_counter[0]:03d}"
    X_eng = get_all_features(train)
    y_check = train['target'].values if 'target' in train.columns else y
    X_test_eng = get_all_features(test)

    hypothesis = "Engineered features significantly improve SVC RBF over raw features"
    print(f"\n--- {eid}: Testing hypothesis ---")
    print(f"  {hypothesis}")
    print(f"  Features: raw ({train.drop(columns=['id','target'] if 'target' in train.columns else ['id']).shape[1]}) "
          f"→ engineered ({X_eng.shape[1]})")

    res = run_experiment(eid, "EXP-005", hypothesis, "full_engineered",
                          "SVC_RBF", MODELS['SVC_RBF'],
                          X_eng.values, y_check, X_test_eng.values, sample_sub,
                          seed=42, n_repeats=2)

    if res and res['oof_accuracy'] > best_score['accuracy']:
        best_score['accuracy'] = res['oof_accuracy']
        best_score['macro_f1'] = res['macro_f1']
        best_score['model'] = 'SVC_RBF'
        best_score['feature_set'] = 'full_engineered'
        best_score['exp_id'] = eid
        save_checkpoint(eid, 'SVC_RBF', res, 'full_engineered',
                        "{C:10, gamma:scale}")
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    experiment_count += 1

    # EXP-007: RandomForest + engineered features
    exp_id_counter[0] += 1
    eid = f"EXP-{exp_id_counter[0]:03d}"
    res = run_experiment(eid, "EXP-006", "RandomForest should benefit from engineered features",
                          "full_engineered", "RandomForest", MODELS['RandomForest'],
                          X_eng.values, y_check, X_test_eng.values, sample_sub,
                          seed=42, n_repeats=2)

    if res and res['oof_accuracy'] > best_score['accuracy']:
        old_acc = best_score['accuracy']
        improvement = res['oof_accuracy'] - old_acc
        if improvement >= 0.002:
            best_score['accuracy'] = res['oof_accuracy']
            best_score['macro_f1'] = res['macro_f1']
            best_score['model'] = 'RandomForest'
            best_score['feature_set'] = 'full_engineered'
            best_score['exp_id'] = eid
            save_checkpoint(eid, 'RandomForest', res, 'full_engineered',
                            "{n_estimators:400, max_depth:14, min_samples_leaf:3}")
            no_improvement_count = 0
        elif res['oof_accuracy'] > old_acc:
            print(f"  Marginal improvement ({improvement:.4f}), keeping but not replacing best")
            no_improvement_count += 1
        else:
            no_improvement_count += 1
    elif res:
        no_improvement_count += 1

    experiment_count += 1

    # EXP-008: XGBoost + engineered features
    try:
        import xgboost as xgb
        exp_id_counter[0] += 1
        eid = f"EXP-{exp_id_counter[0]:03d}"
        model = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, n_jobs=-1, verbosity=0)
        print(f"\n--- {eid}: XGBoost + engineered features ---")
        res = evaluate_model_cv(X_eng.values, y_check, model, "XGBoost",
                                 X_test=X_test_eng.values, seed=42, n_repeats=2)
        log_experiment(eid, "EXP-007", "XGBoost with engineered features",
                        "full_engineered", "XGBoost", str(model.get_params()),
                        "RepeatedStratifiedKFold(5,2)", 42, res,
                        "Ya" if res['oof_accuracy'] > best_score['accuracy'] else "Tidak", "")
        if res['oof_accuracy'] > best_score['accuracy'] + 0.002:
            best_score['accuracy'] = res['oof_accuracy']
            best_score['macro_f1'] = res['macro_f1']
            best_score['model'] = 'XGBoost'
            best_score['exp_id'] = eid
            save_checkpoint(eid, 'XGBoost', res, 'full_engineered', str(model.get_params()))
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        experiment_count += 1
    except ImportError:
        print("  XGBoost not available, skipping")

    # EXP-009: LightGBM + engineered features
    try:
        import lightgbm as lgb
        exp_id_counter[0] += 1
        eid = f"EXP-{exp_id_counter[0]:03d}"
        model = lgb.LGBMClassifier(n_estimators=400, max_depth=8, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, n_jobs=-1, verbose=-1)
        print(f"\n--- {eid}: LightGBM + engineered features ---")
        res = evaluate_model_cv(X_eng.values, y_check, model, "LightGBM",
                                 X_test=X_test_eng.values, seed=42, n_repeats=2)
        log_experiment(eid, "EXP-007", "LightGBM with engineered features",
                        "full_engineered", "LightGBM", str(model.get_params()),
                        "RepeatedStratifiedKFold(5,2)", 42, res,
                        "Ya" if res['oof_accuracy'] > best_score['accuracy'] else "Tidak", "")
        if res['oof_accuracy'] > best_score['accuracy'] + 0.002:
            best_score['accuracy'] = res['oof_accuracy']
            best_score['macro_f1'] = res['macro_f1']
            best_score['model'] = 'LightGBM'
            best_score['exp_id'] = eid
            save_checkpoint(eid, 'LightGBM', res, 'full_engineered', str(model.get_params()))
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        experiment_count += 1
    except ImportError:
        print("  LightGBM not available, skipping")

    # EXP-010: CatBoost + engineered features
    try:
        import catboost as cb
        exp_id_counter[0] += 1
        eid = f"EXP-{exp_id_counter[0]:03d}"
        model = cb.CatBoostClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                       random_seed=42, verbose=0)
        print(f"\n--- {eid}: CatBoost + engineered features ---")
        res = evaluate_model_cv(X_eng.values, y_check, model, "CatBoost",
                                 X_test=X_test_eng.values, seed=42, n_repeats=2)
        log_experiment(eid, "EXP-007", "CatBoost with engineered features",
                        "full_engineered", "CatBoost", str(model.get_params()),
                        "RepeatedStratifiedKFold(5,2)", 42, res,
                        "Ya" if res['oof_accuracy'] > best_score['accuracy'] else "Tidak", "")
        if res['oof_accuracy'] > best_score['accuracy'] + 0.002:
            best_score['accuracy'] = res['oof_accuracy']
            best_score['macro_f1'] = res['macro_f1']
            best_score['model'] = 'CatBoost'
            best_score['exp_id'] = eid
            save_checkpoint(eid, 'CatBoost', res, 'full_engineered', str(model.get_params()))
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        experiment_count += 1
    except ImportError:
        print("  CatBoost not available, skipping")

    # Print final status
    print(f"\n{'='*60}")
    print("PHASE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"Experiments run: {experiment_count}")
    print(f"Best model: {best_score['model']}")
    print(f"Best OOF accuracy: {best_score['accuracy']:.4f}")
    print(f"Best macro F1: {best_score['macro_f1']:.4f}")
    print(f"No improvement count: {no_improvement_count}")
    print(f"\nNext up: Phase 2 — more feature combinations, "
          f"feature selection, Hyperparameter tuning")

    # Save summary
    summary = {
        'phase': 1,
        'experiments_run': experiment_count,
        'best_accuracy': best_score['accuracy'],
        'best_macro_f1': best_score['macro_f1'],
        'best_model': best_score['model'],
        'best_feature_set': best_score.get('feature_set', ''),
        'no_improvement_count': no_improvement_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(EXP_DIR / "reports" / "phase1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
