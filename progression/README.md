# Progression–Cachexia Overlap Permutation Test

This test whether **radiographic progression events** occur within **cachexia (weight-loss) episodes** more often than expected by chance, using a **within-patient permutation null** that preserves each patient’s episode durations and during episode/outside episode labels.

The key output per cancer type is:
- **Observed normalized overlap**: fraction of progression events that fall within spans
- **Permutation null distribution** of the same statistic
- **Two-sided permutation p-value** comparing observed to null

---
For each cancer type :

1. Identify patients with ≥1 progression event.
2. For each patient:
   - Compute **observed overlap**: number of progression events whose `days_since_anchor` falls within any cachexia span (`span == 1`).
   - Generate a **null distribution** by permuting the ordering of that patient’s spans while preserving each span’s duration and cachexia label, then recomputing overlap.
3. Aggregate across patients:
   - `P_norm = (sum observed overlaps across patients) / (total progression events)`
   - `S_perm[k] = (sum permuted overlaps across patients in permutation k) / (total progression events)`
4. Compute permutation p-value:
   - `p = mean( abs(S_perm - mean(S_perm)) >= abs(P_norm - mean(S_perm)) )`

---

## Required inputs

### 1) Spans file (cachexia episode spans)

A table of longitudinal spans per patient. Minimum required columns:

| column | type | description |
|---|---|---|
| `patient_id` | string | de-identified patient identifier |
| `start_day` | integer | start day since anchor (inclusive) |
| `end_day` | integer | end day since anchor (inclusive) |
| `span` | integer (0/1) | 1 = cachexia span, 0 = non-cachexia span |
| `CANCER_TYPE_DETAILED` | string | cancer type label (used to stratify) |

Additional columns are allowed.

**Example rows**
```csv
patient_id,start_day,end_day,span,CANCER_TYPE_DETAILED
P001,0,29,0,Lung Adenocarcinoma
P001,30,85,1,Lung Adenocarcinoma
P001,86,120,0,Lung Adenocarcinoma
P002,0,40,0,Lung Adenocarcinoma
P002,41,110,1,Lung Adenocarcinoma
