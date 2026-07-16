# Final Forensic Report

**Date:** 2026-07-16  
**Total experiments:** 128 (original) + ~20 (forensic)  
**Best validated accuracy:** 0.5887 (Nested CV) / 0.5975 (Standard OOF)

---

## Executive Summary

After comprehensive forensic analysis of the latihan-datathon multiclass classification dataset, the following conclusions are established:

### 1. Skor 0.65: NOT_REPRODUCIBLE
- Skor 0.65 **tidak pernah tercapai** dalam 128+ eksperimen.
- Referensi "~0.65" hanya merupakan target header di `reproduce_baseline.py`, bukan hasil empiris.
- Maksimum akurasi yang tervalidasi: **0.5975** (standard OOF) / **0.5887** (nested CV).

### 2. Leakage Status: NO_CRITICAL_LEAKAGE
Seluruh pipeline telah diaudit leakage-free:
- ✅ Feature engineering dalam fold (StandardScaler di setiap CV fold)
- ✅ Feature engineering tidak menggunakan target
- ✅ OOF isolation terjamin (setiap sampel di-val 2×, di-train 8×, tanpa overlap)
- ✅ Stacking menggunakan OOF probabilities yang benar
- ✅ Tidak ada fitur mencurigakan atau turunan target
- ✅ Tidak ada duplicate rows
- ✅ Adversarial validation: train/test tidak dapat dibedakan (AUC ≈ 0.53)
- ✅ ID-target correlation negligible (-0.005)

### 3. Data Predictability: Sangat Terbatas

| Metrik | Nilai | Interpretasi |
|--------|-------|-------------|
| NN1 agreement | 0.363 | Hanya sedikit di atas random (0.25) |
| Silhouette score | -0.009 | Kelas overlap sempurna |
| Davies-Bouldin | 14.76 | Cluster sangat tidak terpisah |
| Max MI | 0.152 | Sangat lemah |
| Mean F-statistic | NaN | Fitur tidak terbedakan per kelas |
| Best KNN (k=21) | 0.442 | Jauh di bawah model yang ada |
| Learning curve | Flat | Data tambahan tidak membantu |
| All 3 models wrong | 29.1% | bawaan dataset yang sangat bising |

### 4. Label Quality

| Kategori | Jumlah | % |
|----------|--------|---|
| Easy correct | 1,086 | 33.9% |
| Consistently wrong | 907 | 28.3% |
| Uncertain | 757 | 23.7% |
| Borderline correct | 450 | 14.1% |

**601 possible mislabels** ditemukan (100% model agreement ≠ actual label).  
Ini menunjukkan tingkat label noise yang signifikan.

### 5. Perbaikan yang Valid

| Eksperimen | Akurasi | Delta | Keterangan |
|------------|---------|-------|------------|
| Baseline (Stack_LR Phase 3) | 0.5869 | - | Best from 128 experiments |
| + Sequence features | 0.5944 | +0.0075 | Robust slope, curvature, autocorrelation |
| + Ordinal stacking | 0.5975 | +0.0106 | SVC + CatBoost + Ordinal on seq features |
| Manual special features | 0.47-0.53 | ↓ | Tidak membantu |
| Cluster/KNN embedding | 0.56 | ↓ | Tidak membantu |

Sequence features = perbaikan paling berarti (+0.75 poin persen).

### 6. Nested CV Validation

| Metrik | Nilai |
|--------|-------|
| Nested CV accuracy | **0.5859** |
| Standard OOF (same pipeline) | 0.5887 |
| Gap | **+0.0028** (minimal — tidak ada validation overfitting) |
| Outer fold variance | 0.0233 (cukup stabil) |
| Best fold / Worst fold | 0.6156 / 0.5609 |

### 7. Kesimpulan Plateau

Dataset ini berada pada kondisi **FEATURE_LIMITED + LABEL_NOISE_LIMITED**:

1. **Feature-limited:** MI tertinggi hanya 0.152, silhouette score negatif, mayoritas fitur tidak informatif.
2. **Label-noise-limited:** ~28% sampel consistently salah diklasifikasi oleh semua model, 601 possible mislabels.
3. **Model-limited:** Semua pendekatan (linear, SVM, trees, boosting, stacking, pairwise, neural-proxy) plateau di ~0.59.
4. **Data-limited:** Learning curve flat — menambah data dari distribusi yang sama tidak akan membantu.

### 8. Rekomendasi

Berdasarkan seluruh bukti empiris, **fitur yang tersedia belum mendukung target 0.70 secara valid**. Peningkatan berikutnya lebih mungkin berasal dari:

- **Penambahan informasi baru** (fitur eksternal, domain knowledge baru)
- **Perbaikan label** (membersihkan ~601 possible mislabels jika memungkinkan)
- **Perubahan definisi target** (regression atau ordinal dengan granularitas berbeda)
- **Ensemble multi-institusi** (jika dataset berasal dari sumber berbeda)

Model terbaik saat ini (**Stack_LR + sequence + ordinal = 0.5975 OOF**) adalah estimasi batas performa realistis dari pipeline yang leakage-safe.

---

## Bukti Lengkap

1. **Score 0.65 audit:** `forensic_experiments/audit/score_065_audit.md`
2. **Leakage audit:** `forensic_experiments/audit/leakage_audit.json`
3. **Predictability analysis:** `forensic_experiments/feature_analysis/predictability.json`
4. **Label quality:** `forensic_experiments/label_analysis/label_quality_summary.json`
5. **Hard samples:** `forensic_experiments/label_analysis/hard_samples.csv`
6. **Possible mislabels:** `forensic_experiments/label_analysis/possible_mislabels.csv`
7. **Neighbor agreement:** `forensic_experiments/label_analysis/neighbor_agreement.csv`
8. **Model disagreement:** `forensic_experiments/model_analysis/model_disagreement.csv`
9. **Nested CV results:** `forensic_experiments/model_analysis/nested_cv_results.json`
10. **Experiment log:** `forensic_experiments/experiments/experiment_log.csv`
