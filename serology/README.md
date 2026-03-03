# Serology GLMM 

This folder contains the R script(s) used to test whether serologic lab values are associated with episode status using mixed-effects logistic regression (GLMM).

For each **cancer type** and each **lab test**, we fit a logistic mixed-effects model where:

- **Outcome:** `span` (episode status for a span/interval; 0/1)
- **Predictors:** the lab value (`result`), age at anchor (`age_at_diag`), and sex (`sex`)
- **Random effect:** patient (`PATIENT_ID`) to account for repeated measurements within a patient

The output is a table of per-(cancer, lab) effect sizes (OR), 95% CIs, and p-values, with BH-FDR correction across tests.

---

### Required input table: `spans_labtests` (CSV)
A wide-format table where each row is a cachexia consensus-defined weight loss “span” observation with labs aligned to that span.

Minimum required columns:

- `PATIENT_ID` — patient identifier (can be de-identified)
- `CANCER_TYPE_DETAILED` — cancer type label (used for stratification)
- `span` — cachexia status for the span (0/1)
- `age_at_diag` — numeric age at anchor/diagnosis (in years)
- `sex` — sex label (e.g., `FEMALE`, `MALE`)
- Lab columns — one column per lab test (numeric), e.g. `Albumin`, `HGB`, `WBC`, `Neut`, `ALK`, etc.

Example rows (illustrative):

| P_ID | CANCER_TYPE_DETAILED      | span  |   AGE_AT_DIAGNOSIS  |   SEX  | Albumin | HGB | WBC | Neut | ALK |
|------|---------------------------|------|----------------------|--------|---------|-----|-----|------|-----|
| P001 | Lung Adenocarcinoma       | 1    | 62.4                 | FEMALE | 3.1     | 10.2| 11.8| 8.9  | 210 |
| P001 | Lung Adenocarcinoma       | 0    | 62.4                 | FEMALE | 3.8     | 12.9| 7.4 | 4.2  | 120 |
| P002 | Colon Adenocarcinoma      | 1    | 55.0                 | MALE   | 2.9     | 9.8 | 13.1| 10.0 | 180 |
| P003 | Prostate Adenocarcinoma   | 0    | 70.1                 | MALE   | 4.0     | 13.5| 6.5 | 3.8  | 95  |

Notes:
-  Each lab’s model uses only rows where that lab is non-missing.

### Model
We fit a logistic mixed-effects model (GLMM) with a random intercept for patient:

If both sexes are present in the cancer-specific subset:
span ~ result + AGE_AT_DIAG + sex + (1 | PATIENT_ID)

If only one sex exists in that subset (e.g., sex-specific cancers), `sex` is dropped:
span ~ result + AGE_AT_DIAG +  (1 | PATIENT_ID)

### Stratification
Models are fit separately for each `CANCER_TYPE_DETAILED`. Typically, cancer types are filtered to a minimum cohort size (e.g., ≥500 patients), but this threshold is configurable upstream.

### Multiple testing correction
P-values are adjusted across tests using Benjamini–Hochberg (BH-FDR). The adjusted p-value is reported as `adjusted_p`.

## Outputs

### Results table (CSV)
One row per (cancer type × lab test), with:

- `test` — lab test name (column name)
- `cancer_type` — cancer label
- `estimate` — odds ratio (OR = exp(beta))
- `logor` — beta coefficient on log-odds scale
- `lower_ci`, `upper_ci` — 95% CI for OR
- `p` — Wald p-value for the lab coefficient
- `adjusted_p` — BH-FDR adjusted p-value across tests

## Required R packages
The core pipeline uses:
- `data.table` (I/O and fast handling)
- `dplyr` (filtering / mutation)
- `lme4` (GLMM fitting)
