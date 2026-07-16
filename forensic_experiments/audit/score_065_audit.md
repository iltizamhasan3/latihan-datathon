# Audit Skor 0.65

## Status: NOT_REPRODUCIBLE

### Ringkasan
Setelah memeriksa seluruh 128 eksperimen, 8 fase, dan semua file laporan:
- **Skor tertinggi yang valid**: 0.5869 (Phase 3 - Stack_LR)
- **Tidak ada eksperimen** yang pernah mencatat skor >= 0.59
- **Tidak ada eksperimen** yang pernah mencatat skor >= 0.60
- **Tidak ada eksperimen** yang pernah mencatat skor >= 0.65

### Asal-usul Klaim 0.65
Satu-satunya referensi ke "0.65" adalah di file header :


Ini adalah **target yang ditulis sebelum baseline pertama dijalankan**, bukan hasil yang tercapai.
Setelah baseline dijalankan, SVC RBF raw hanya mencapai 0.4748 — jauh dari 0.65.

### Verifikasi Lengkap

| Sumber | Hasil |
|--------|-------|
| experiment_log.csv (128 baris) | Max = 0.5869 |
| phase1_summary.json | Max = 0.5156 |
| phase6_summary.json | Max = 0.5869 |
| final_report.json | Best = 0.5869 |
| data_audit.json | N/A (audit) |
| best_config.json | Best = 0.5869 |
| Semua file submission | Tidak ada yang >= 0.59 |

### Kesimpulan
Skor 0.65 **tidak pernah ada**. Klaim tersebut tidak berdasar dan seharusnya tidak digunakan sebagai referensi untuk eksperimen selanjutnya. Baseline yang valid adalah **0.5869**.
