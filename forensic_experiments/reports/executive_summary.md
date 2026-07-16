# Executive Summary — Datathon Forensic Analysis

## Status: ✅ PLATEAU CONFIRMED — DATA LIMITED

### Best Validated Accuracy: 0.5887 (Nested CV) / 0.5975 (Standard OOF)

### Timeline
| Event | Detail |
|-------|--------|
| 128 original experiments | Best = 0.5869 (Stack_LR) |
| + Sequence features | 0.5944 (+0.0075) |
| + Ordinal stacking | **0.5975** (+0.0106) |
| Nested CV validation | 0.5859 (gap = 0.0028 ✅ stable) |

### Skor 0.65: NOT_REPRODUCIBLE
Tidak ada bukti bahwa skor 0.65 pernah tercapai. Referensi tersebut hanya target header.

### Key Limiting Factors
1. **Label noise**: ~601 possible mislabels, 28% samples consistently wrong
2. **Feature weakness**: Max MI = 0.152, silhouette = -0.009
3. **Class overlap**: Classes 0↔1 and 2↔3 frequently confused
4. **Flat learning curve**: More data of same distribution won't help

### Recommendation
Target 0.70 tidak didukung oleh informasi fitur saat ini.
Peningkatan membutuhkan fitur baru, perbaikan label, atau perubahan definisi target.
