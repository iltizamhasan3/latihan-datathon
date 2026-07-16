# Data Identity Report

## File Information
| Property | Train | Test |
|----------|-------|------|
| SHA256 | `8a103527e554ed5c2c3b3b25d0cf188b0ed015877965a6e4df8df46395c53918` | `e1adb54508dfeefd2f78c944e8a3f825995ca807f849645347dc7a889a4c8e11` |
| Rows | 3200 | 800 |
| Columns | 43 (incl target) | 42 |
| Duplicate rows | 0 | 0 |
| Near-duplicates | 0 | 0 |
| Train-test row overlap | 0 | - |

## Target Distribution
| Class | Count | % |
|-------|-------|---|
| 0 | 813 | 25.41 |
| 1 | 796 | 24.88 |
| 2 | 784 | 24.50 |
| 3 | 807 | 25.22 |

## Row Order
- ID range: 0-3999 (train), 4000-4799 (test)
- ID monotonic: True
- ID gaps: 659 gaps of 2-4 → train is a sampled subset (not contiguous)
- Target autocorrelation lag1: 0.04 (near zero)
- Target run changes: 2353/3200 (expected 2400 for random 4-class)
- No structural order pattern

## Kelas Analysis (IMPORTANT)
- **786 unique classes** in train, 492 in test
- **483/492 test classes overlap with train** (98% overlap)
- Avg 4.1 students per class (range 1-12)
- Some classes show strong target homogeneity:
  - Class 4: 3 samples, all class 2
  - Class 108: 3 samples, all class 3
  - Class 169: 5 samples, all class 0
  - Class 204: 3 samples, all class 3
  - Class 362: 3 samples, all class 3
  - Class 428: 3 samples, all class 0
  - Class 505: 3 samples, all class 0
  - Class 630: 3 samples, all class 3
  - Class 632: 4 samples, all class 0
  - Class 706: 4 samples, all class 0
  - Class 285: 8 samples, 3→5 (biased class 3)
  - Class 491: 8 samples, 1→1→1→5 (biased class 3)
- **Group structure is strong** → kelas-level aggregation features could be high-value

## Feature Types
| Feature Group | Count | Range | Decimal Precision |
|---|---|---|---|
| Weekly scores (nilai_minggu_*) | 12 | -15.2 to +14.7 | 1 decimal |
| Activity (aktivitas_hari_*) | 16 | 0.9 - 104.0 | 1 decimal |
| Soft skills (motivasi, kedisiplinan) | 2 | -5.2 to +6.3 | 2 decimals |
| Tasks (selesai, diberikan) | 2 | 0 - 120 | 0 decimals |
| Kelas | 1 | 0-799 | 0 |
| urutan_ujian | 1 | 0.0005 - 0.9996 | 4 decimals |
| skor_tryout | 1 | 21.7 - 96.3 | 1 decimal |
| Behavioral (jarak, ekstrakurikuler, kehadiran, literasi, saudara, minat) | 6 | -3.6 to +4.9 | 2 decimals |

## Key Observations
1. **Kelas is the strongest structure**: 483/492 test samples share kelas with train → group-level features are viable
2. **No leakage via duplicates or ID**
3. **Target is well-balanced** → accuracy is the right metric (0.25 baseline)
4. **Possible synthetic data**: Behavioral features centered near 0, bounded between -5 and +5 suggest they may be normalized/random effects
5. **ID gaps confirm subset sampling**: Not all students in the original population are included

## Potential External Data / Competitor Advantage
- 483/492 test kelas in train → could allow per-class statistics
- If other competitors used `kelas` as group-level features with per-class aggregation, they could extract additional signal
- The `kelas` column may be the key to reaching 0.65+
