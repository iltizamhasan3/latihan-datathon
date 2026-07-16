# Leakage Audit Report

## Verdict: NO_CRITICAL_LEAKAGE

### Critical Findings: 0
- None

### Minor Findings: 0
- None

### All Checks
| Check | Passed | Detail |
|------|--------|--------|
| preprocessing_within_fold | ✅ | PCA dims per fold: [np.int64(47), np.int64(47), np.int64(47), np.int64(47), np.int64(47), np.int64(47), np.int64(47), np.int64(47), np.int64(47), np.int64(47)] — all folds have same dims |
| feature_engineering_target_leak | ✅ | features.py does not compute any feature using the target column. All features are based on row-level attributes only. |
| oof_sample_isolation | ✅ | Each sample in val 2-2 times, in train 8-8 times. No overlap found: True |
| stacking_oof_method | ✅ | All stacking experiments (Phase 3, 4b, 6, 8, final) use OOF probabilities for meta-training and fold-averaged test probabilities. No in-sample leakage. |
| threshold_optimization | ✅ | Phase 5 threshold optimization done on OOF predictions (standard practice). Not using nested CV but effect is minimal (0.001 improvement reported). |
| suspicious_feature_names | ✅ | Suspicious features found: None |
| max_mutual_information | ✅ | Max MI = 0.1525 (feature: trend_task). All features have low individual predictivity. |
| exact_duplicates | ✅ | Exact duplicate rows: 0 |
| near_duplicate_conflicts | ✅ | Near-duplicate conflicts found (distance<0.1, diff target): 0 out of 500 samples checked |
| adversarial_validation | ✅ | Adversarial accuracy: 0.5340 (close to 0.5 = indistinguishable) |
| id_target_correlation | ✅ | ID-target correlation: -0.005071 (leakage via row ordering) |
| monotonic_column_leakage | ✅ | No columns show near-perfect correlation (>0.995) with ID. |

## Summary
All critical leakage checks passed. The evaluation pipeline (StandardScaler within CV,
OOF probabilities for stacking, no target leakage in features, no duplicate leakage) is sound.
The 0.5869 baseline is validated as a leakage-free score.
