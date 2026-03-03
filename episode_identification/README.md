# Episode identification v2 (`episode_identification_v2.py`)

This script identifies episode-resolved weight-loss events from longitudinal BMI trajectories. It smooths BMI using an exponentially weighted moving average (EWMA), detects candidate cachexia episodes using a fixed 180-day look-ahead rule, merges overlapping episode detections, assigns recovery (optional), applies duration/weight-loss quality control (QC), and filters episodes that overlap edema-like “reversible rapid gain” windows in log-BMI space.

---

## Summary of workflow
1. **Load inputs**
   - Cohort metadata (`dx_cohort_metadata_<DATE_STAMP>.csv`)
   - Longitudinal BMI table (`bmi_final_<DATE_STAMP>.csv`)
2. **Minimal filtering**
   - Restrict BMI measurements to those **on/after anchor** (`days_since_anchor >= 0`)
   - De-duplicate to one measurement per patient per day (keeps lowest BMI when duplicates exist)
3. **EWMA smoothing**
   - For each patient, compute `smoothed_BMI` using EWMA with smoothing factor `ALPHA` (default 0.2)
   - Compute `log_smoothed_BMI = log(smoothed_BMI)`
4. **Episode detection (WL5/WL10/WL15)**
   - For each patient and each baseline day `t0`, look forward exactly **180 days**
   - Declare an “episode candidate” if `log_smoothed_BMI` drops below baseline by:
     - `log(1 - wl_frac)` where `wl_frac ∈ {0.05, 0.10, 0.15}`
     - If the “BMI<20 rule” is enabled, use **2% WL** (`log(0.98)`) when baseline `smoothed_BMI < 20`
   - Record start/onset/end dates and BMI values
   - Merge overlapping detections into consolidated episodes
   - Optionally compute recovery as the first time after episode end that `log_smoothed_BMI > end_log_BMI + log(1.05)` (≥5% gain relative to the end BMI)
5. **QC filtering**
   - Keep episodes with:
     - `episode_duration >= MIN_DUR` (default 15 days)
     - `weight_loss >= MIN_WL_PCT` (default 2%), computed from smoothed BMI at start and end
6. **Edema QC**
   - Identify edema-like windows in each patient as **reversible rapid gains** in log-BMI:
     - baseline → increase by ≥ `log(1 + UP_FRAC)` (default 5%) → return to ≤ `log(1 + RETURN_FRAC)` (default 2%)
     - within `EDEMA_WINDOW_DAYS` (default 30 days)
   - Merge overlapping edema windows
   - Remove QC-valid episodes that overlap any edema window

---

## Inputs
### 1) Cohort metadata
**Default filename:** `data/dx_cohort_metadata.csv`
Required columns:
- `MRN`
- `anchor_final` (datetime; tumor diagnosis date)

### 2) Longitudinal BMI table
**Default filename:** `data/bmi_final_<DATE_STAMP>.csv`
Required columns:
- `MRN`
- `datetime` (datetime of BMI measurement)
- `bmi` (numeric BMI)
- `anchor_final` (tumor diagnosis date)

Notes:
- BMI records with missing `MRN`, `datetime`, `bmi`, or `anchor_final` are dropped.
- BMI is filtered to measurements occurring on/after anchor.
---
## Outputs
### Other
- `smoothed_bmi_all_patients_<RUN_STAMP>_alpha<ALPHA>.csv`

### Episode detection (pre-QC; per WL threshold)
- `df_episodes_all_precomp_WL5_<RUN_STAMP>.csv`
- `df_episodes_all_precomp_WL10_<RUN_STAMP>.csv`
- `df_episodes_all_precomp_WL15_<RUN_STAMP>.csv`

### QC outputs (duration + WL% filters; per WL threshold)
- `episode_summary_valid_<WL>_<RUN_STAMP>_dur<MIN_DUR>_wl<MIN_WL_PCT>.csv`
  - Includes all patients (left-joined denominator), with `has_cachexia_valid` indicator
- `valid_episodes_only_<WL>_<RUN_STAMP>_dur<MIN_DUR>_wl<MIN_WL_PCT>.csv`
  - Episode-only table after QC

### Edema windows
- `edema_windows_<RUN_STAMP>_LOG_w<EDEMA_WINDOW_DAYS>_up<UP%>_ret<RET%>.csv`

### QC + edema-filtered outputs (per WL threshold)
- `episode_summary_valid_<WL>_<RUN_STAMP>_dur<MIN_DUR>_wl<MIN_WL_PCT>_edemaQC_LOG_w<EDEMA_WINDOW_DAYS>_up<UP%>_ret<RET%>.csv`
  - Includes all patients, with `has_cachexia_valid_edemaQC` indicator
- `valid_episodes_only_<WL>_<RUN_STAMP>_dur<MIN_DUR>_wl<MIN_WL_PCT>_edemaQC_LOG_w<EDEMA_WINDOW_DAYS>_up<UP%>_ret<RET%>.csv`
  - Episode-only table after QC + edema filtering
---

## Configuration

### Path variables
- `CACHEXIA_PROJECT_ROOT` (default: current working directory)
- `CACHEXIA_INPUTS_DIR` (default: `${PROJECT_ROOT}/data`)
- `CACHEXIA_RESULTS_DIR` (default: `${PROJECT_ROOT}/results`)
- `CACHEXIA_PLOTS_DIR` (default: `${PROJECT_ROOT}/plots`)

### Input file variables
- `CACHEXIA_DATE_STAMP` (default: `YYYYMMDD`)
- `CACHEXIA_DX_FP` (default: `${INPUTS_DIR}/dx_cohort_metadata.csv`)
- `CACHEXIA_BMI_FP` (default: `${INPUTS_DIR}/bmi_final.csv`)

### Detection parameters
- `CACHEXIA_EWMA_ALPHA` (default: `0.2`)
- `CACHEXIA_TIME_COL` (default: `days_since_anchor`)
- `CACHEXIA_BMI_LOG_COL` (default: `log_smoothed_BMI`)
- `CACHEXIA_MIN_DUR` (default: `15`)
- `CACHEXIA_MIN_WL_PCT` (default: `2`)
- `CACHEXIA_EDEMA_WINDOW_DAYS` (default: `30`)
- `CACHEXIA_EDEMA_UP_FRAC` (default: `0.05`)
- `CACHEXIA_EDEMA_RETURN_FRAC` (default: `0.02`)

### BMI<20 rule (2% WL when baseline smoothed BMI < 20)
Enable by setting either:
- `CACHEXIA_EPISODE_MODE=bmi20_rule`, or
- `CACHEXIA_BMI20_RULE=1`
---

## Run pipeline
### Minimal run (inputs in `data/`)
```bash
export CACHEXIA_DATE_STAMP=20260126
python scripts/episode_identification_v2.py
