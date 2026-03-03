# A health-system-scale, episode-resolved multimodal atlas of cancer cachexia

This repository contains the core analysis code used to construct an episode-resolved cachexia atlas from longitudinal BMI trajectories and to perform downstream multimodal analyses (genomics, serology, and progression overlap). 

**Start with:** start with the `walkthrough/` folder. It provides a step-by-step guide to episode detection and the key intermediate files produced by the cachexia identification pipeline.

---
## Repository

- `walkthrough/`  
  Step-by-step walkthrough of episode detection and how to reproduce intermediate outputs.  
  **Start here first.**

- `episode_identification/`  
  Episode-resolved cachexia detection from longitudinal BMI:
  EWMA smoothing → 180-day window detection → episode merging → recovery labeling → duration/weight-loss QC → edema QC.

- `genomics/`  
  Genomic association and prediction models aligned to cachexia time-to-event:
  univariate Cox → multivariable Cox → prediction risk model (calibration + time-dependent AUC).

- `serology/`  
  Serologic signatures aligned to cachexia windows, including sensitivity analyses.

- `progression/`  
  Progression alignment and overlap analyses 
---


## Data requirements

Input files are not included in the repository.
At minimum, analyses assume the availability of:

### Longitudinal BMI table
A table of BMI measurements over time per patient, including:
- patient identifier (e.g., `MRN`)
- measurement datetime (`datetime`)
- BMI value (`bmi`)
- anchor date used to standardize time (e.g., diagnosis/anchor date)

### Cohort metadata table
A patient-level metadata table including:
- patient identifier (e.g., `MRN`)
- anchor date (e.g., `anchor_final` or `Tumor Diagnosis Date`)
- cancer type label (e.g., `CANCER_TYPE_DETAILED`)
- other clinical covariates depending on the analysis module (e.g., stage, sex, etc.)

### Additional modality-specific tables
Depending on the module:
- mutation table keyed to a genomics identifier (e.g., `DMP_ID`) for `genomics/`
- longitudinal lab values for `serology/`
- progression timepoints/labels for `progression/`

Each subfolder contains a module-specific README describing exact expected inputs/columns and produced outputs.
---


## Outputs 
- **Episode detection outputs (`episode_identification/`):**
  - smoothed BMI trajectories
  - episode tables per threshold (WL5/WL10/WL15)
  - QC-filtered episode summaries (patient-level + episode-level)
  - edema-filtered episode summaries
  - incidence summary outputs and QC diagnostics

- **Genomics outputs (`genomics/`):**
  - univariate per-cancer Cox results across genes
  - multivariable per-cancer Cox models including clinical covariates
  - nomogram-style evaluation plots (calibration and time-dependent ROC/AUC)

- **Serology outputs (`serology/`):**
  - cachexia-aligned lab models and summary tables
  - sensitivity analyses across detection parameters and cachexia definitions

- **Progression outputs (`progression/`):**
  - cachexia–progression overlap summaries
  - stage-stratified and sensitivity analyses where relevant

---
