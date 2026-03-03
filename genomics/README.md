
# Genomics association + prediction pipeline (cachexia time-to-event)

This repository contains three scripts to (i) test per-gene associations with time-to-cachexia (univariate Cox), (ii) fit per-cancer multivariable Cox models using significant genes plus clinical covariates, and (iii) evaluate a risk model for a cancer type (bootstrap C-index, calibration, and time-dependent AUC).

Scripts (run in order):

1. `scripts/univariate_genomics.py`
2. `scripts/multivariable_genomics.py`
3. `scripts/build_nomogram.py`

---

### Identifiers
- `MRN` is used to merge BMI/episodes to cohort metadata.
- `DMP_ID` is used to merge cohort metadata to mutation calls.
- Scripts truncate `DMP_ID` to the first 9 characters (`str[:9]`) to match keying across tables.

### Time units
- All time-to-event values are in **days**.
- Nomogram evaluation uses ~1/2/3-year horizons defined as 365/730/1095 days, mapped to the nearest available timepoints in the predicted survival function.

### Default outputs
All outputs are written under:
- `results_mutations/results_mutation_cac_frozen_<WL_TAG>/`

---
## Inputs
### 1) Cohort metadata  
**Filename:** `dx_cohort_metadata.csv`
Used by all steps.
Required columns:
- `MRN`
- `DMP_ID`
- `CANCER_TYPE_DETAILED`
- `PLA_LAST_CONTACT_DTE`
- `PT_DEATH_DTE`
- `PT_BIRTH_DTE` *(required for multivariable + nomogram)*
- `GENDER` *(required for multivariable + nomogram)*
- `SAMPLE_TYPE` *(required for multivariable + nomogram)*
- `CVR_TMB_SCORE` *(required for multivariable + nomogram)*
- `STAGE_CDM_DERIVED_GRANULAR` *(required for multivariable + nomogram)*
- Either:
  - `Tumor Diagnosis Date`
Only required for specific cancer-type:
- `MSI_TYPE` (used for CRC / stomach / uterine endometrioid)
- `Sidedness` (CRC only, if used)

### 2) Mutation table  
**Filename:** `oncogenic_table.csv`

Used by all steps (directly or indirectly).

Required columns:

- First column: an identifier that will be renamed to `DMP_ID`
- Subsequent columns: genes (numeric 0/1 or counts; values > 1 are clipped to 1)

Assumptions:

- `DMP_ID` is converted to string and truncated to 9 characters to match metadata keying.

### 3) BMI longitudinal table  
**Filename:** `bmi.csv`

Used by multivariable + nomogram steps to compute baseline BMI near diagnosis.
Required columns:
- `MRN`
- `datetime` (date or datetime string)
- `bmi` (numeric)

### 4) ECOG/KPS timeline table  
**Filename:** `ecog_kps_timeline.tsv` (tab-separated)

Used by multivariable + nomogram steps to compute baseline ECOG/KPS.

Required columns:

- `PATIENT_ID` (renamed/treated as `DMP_ID` for merging)
- `START_DATE` (numeric timeline; baseline is closest to 0)
- `ECOG_KPS`
  // Baseline ECOG/KPS is defined as the measurement closest to `START_DATE == 0`.

### 5) Survival dataset (from by Step 01)  
**Filename:** `cachexia_data_survival_mutation_cac_frozen_<WL_TAG>.csv`

This is the analysis-ready dataset used by Steps 02 and 03.
Minimum required columns:
- `MRN`
- `DMP_ID`
- `CANCER_TYPE_DETAILED`
- `time_to_cachexia` (days)
- `cachexia_event` (0/1)
- `Tumor Diagnosis Date`
- `PT_BIRTH_DTE`
- `GENDER`
- `SAMPLE_TYPE`
- `CVR_TMB_SCORE`
- `STAGE_CDM_DERIVED_GRANULAR`
Only required for specific models:
- `MSI_TYPE`
- `Sidedness`

---

## Step 01 — Univariate genomics tests  
**Script:** `scripts/univariate_genomics.py`
-Tests per-gene associations with time-to-first cachexia, stratified by cancer type, using univariate Cox proportional hazards models.

Per cancer type:
1. Builds or loads an analysis-ready dataset with:
   - `time_to_cachexia` (days to first cachexia event)
   - `cachexia_event` (event indicator)
   - merged binary mutation calls
2. Filters to:
   - cancer types with sufficient sample size (e.g., >200 patients; per script)
   - genes with mutation prevalence >5% within that cancer type (per script)
3. Fits univariate Cox models:
   - `time_to_cachexia ~ gene`
4. Adjusts p-values using Benjamini–Hochberg FDR across all cancer–gene tests.
5. Generates a volcano plot of effect size vs significance.
6. (If enabled in the univariate script) generates CIF plots (Aalen–Johansen) for significant cancer–gene pairs.

### Inputs
- `dx_cohort_metadata_<DATE_STAMP>.csv`
- `oncogenic_table.csv`
- episode-derived tables with `time_to_cachexia` - first episode

### Outputs
Under: `results_mutations/results_mutation_cac_frozen_<WL_TAG>/`

other outputs:
- `cachexia_data_survival_mutation_cac_frozen_<WL_TAG>.csv`
- `mutation_cox_cachexia_results_ccr_updated_<WL_TAG>.csv` (includes `p_adj`)
- `plots/volcano_plot_*.pdf`
- `plots_cif/`: CIF PDFs per significant cancer–gene

---

## Step 02 — Multivariable genomics models  
**Script:** `scripts/multivariable_genomics.py`
-Fits per-cancer multivariable Cox models using significant genes from Step 01 plus clinical covariates.

For each selected cancer type:
1. Loads `cachexia_data_survival_mutation_cac_frozen_<WL_TAG>.csv` from Step 01.
2. Computes covariates:
   - `age_at_diagnosis` from diagnosis date and birth date (scaled by 10 years in the script)
   - baseline `start_BMI` from BMI measurement closest to diagnosis (scaled by 5 in the script)
   - baseline `ECOG_KPS` from ECOG timeline closest to `START_DATE == 0`
3. Selects significant genes from Step 01 (default: FDR < 0.1) for that cancer type.
4. Fits a multivariable Cox model of the form:

   - `time_to_cachexia ~ sig_genes + age_at_diagnosis + GENDER + SAMPLE_TYPE + CVR_TMB_SCORE + STAGE_CDM_DERIVED_GRANULAR + ECOG_KPS + start_BMI`

   Additional terms may be included for certain cancer types, depending on columns available:
   - `MSI_TYPE` (CRC / stomach / uterine endometrioid)
   - `Sidedness` (CRC only, if present and used)
5. Saves coefficient tables for each cancer type and applies FDR correction across covariates within each model output table.

### Inputs
- `cachexia_data_survival_mutation_cac_frozen_<WL_TAG>.csv`
-`mutation_cox_cachexia_results_ccr_updated_<WL_TAG>.csv`
- `bmi.csv`
- `ecog_kps_timeline.tsv`
- `dx_cohort_metadata.csv` 

### Outputs
- `<CancerType>.csv` (lifelines Cox summary table + `p_adj`)

---

## Step 03 — Build nomogram-style evaluation  
**Script:** `scripts/build_nomogram.py`
-Evaluates a cancer-type–specific risk model by selecting significant multivariable covariates, refitting a Cox model, and computing discrimination and calibration.

For one cancer type:
1. Loads the Step 01 survival dataset:
   - `cachexia_data_survival_mutation_cac_frozen_<WL_TAG>.csv`
2. Loads the Step 02 multivariable results table for the selected cancer type:
   - `multivariate_<WL_TAG>_<DATE_STAMP>/cancer_types/<CancerType>.csv`
3. Selects significant covariates (default: FDR < 0.1).
4. Refits a Cox model using only those covariates.
5. Evaluates performance:
   - **Bootstrap C-index**: resamples the dataset and reports mean + 95% CI.
   - **Calibration**:
     - 50/50 train/test split
     - predicted cachexia incidence at ~1/2/3 years using nearest available survival-function timepoints
     - risk quartiles (quantile bins)
     - observed incidence per bin estimated with Kaplan–Meier, with bootstrap CIs
   - **Time-dependent AUC/ROC**:
     - uses scikit-survival cumulative dynamic AUC at 1/2/3-year horizons
     - plots time-dependent ROC curves with AUC labels

### Inputs
- Step 01 output:
  - `cachexia_data_survival_mutation_cac_frozen_<WL_TAG>.csv`
- Step 02 output:
  - `multivariate_<WL_TAG>_<DATE_STAMP>/cancer_types/<CancerType>.csv`
- `bmi.csv`
- `ecog_kps_timeline.tsv`
- `dx_cohort_metadata_<DATE_STAMP>.csv`

### Outputs
Outputs:
- `calibration_half_split_<CancerType>_<WL_TAG>.pdf`
- `auc_half_split_<CancerType>_<WL_TAG>.pdf`

---

