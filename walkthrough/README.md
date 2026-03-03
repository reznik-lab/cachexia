
# Mock data for episode-identification walkthrough

This folder contains **small, de-identified mock inputs** as input for the walkthrough.  

## Walkthrough data
- `example_patient_deid.csv`  
  A single mock patient BMI time series (de-identified). 

## CSV format
The walkthrough expects the input to include the following columns:

- `patient_id` (string)  
  De-identified patient identifier.

- `days_since_anchor` (integer)  
  Days since the anchor date (anchor = diagnosis day in the walkthrough).  
  Example: `0` = diagnosis/anchor, `30` = 30 days after.

- `bmi` (numeric)  
  BMI value recorded at that timepoint.
### Example (rows)

patient_id,days_since_anchor,bmi  
EXAMPLE_PATIENT_001,0,24.8  
EXAMPLE_PATIENT_001,30,24.1  
EXAMPLE_PATIENT_001,60,23.7  
EXAMPLE_PATIENT_001,90,22.9  


# Episode identification walkthrough (mock example)

This walkthrough demonstrates, end-to-end, how we identify **weight-loss–defined cachexia episodes** from longitudinal BMI trajectories using a **small, de-identified mock patient**. It is intended as a transparent, reproducible example of the episode-detection logic used in the main analysis.

## Contents
- `walkthrough/episode_identification_walkthrough.ipynb`  
  The step-by-step notebook that loads mock data, visualizes BMI, applies smoothing, runs episode detection, and plots detected episodes.
- `walkthrough/example_patient_deid.csv`  
  A small de-identified BMI time series used as the notebook input.

## Quick start
1. Open `walkthrough/episode_identification_walkthrough.ipynb` locally in Jupyter.
2. Run cells top-to-bottom.

## Input data format
Thisreads a CSV with the following required columns:
- `patient_id` (string): de-identified patient identifier
- `days_since_anchor` (integer): days since the anchor date (anchor = diagnosis day in this walkthrough)
- `bmi` (numeric): BMI at that timepoint

See above for additional formatting notes and a toy example.

## What the walkthrough shows
This illustrates the main steps used in episode identification:

1. **Load and QC the BMI series**  
   - subset to one patient  
   - sort by `days_since_anchor`  
   - drop missing BMI rows  

2. **Visualize raw BMI**  
   - plot BMI vs. standardized time  
   - include a reference line at BMI = 20 (used in the WL2/BMI<20 rule)

3. **Smooth the BMI trajectory (EWMA)**  
   - apply EWMA smoothing to reduce measurement noise  
   - visualize raw vs. smoothed BMI

4. **Detect candidate weight-loss windows (sliding window)**  
   - scan forward in time using a fixed window length (e.g., 180 days)  
   - flag windows meeting the weight-loss criterion:
     - **≥5% weight loss over 6 months**, OR  
     - **≥2% weight loss if BMI < 20** (rule used in the consensus definition)

5. **Merge overlapping/adjacent windows into episodes**  
   - overlapping windows are merged into a single contiguous episode interval  
   - the merged interval defines the episode’s **start** and **end** boundaries

6. **Validity QC (candidate-window QC)**
After candidate windows are identified, the pipeline applies **validity checks** to remove windows that are not interpretable or are likely spurious. Examples of validity criteria include:
- enforcing minimum episode duration after merging (e.g., ≥15 days)
- enforcing a minimum absolute weight-loss magnitude consistent with the definition

7. **Edema QC (rapid fluctuation QC)**
Because short-term fluid shifts can mimic true tissue loss/gain, the pipeline includes an **edema-based QC filter** to exclude episodes dominated by rapid BMI changes. Conceptually, this step:
- searches around candidate windows/episodes for **rapid up/down fluctuations** over short time horizons
- flags trajectories suggestive of edema or fluid-related noise rather than sustained weight loss
- removes affected episodes (or affected windows prior to merging
  
## Citation / terminology
This walkthrough demonstrates a **weight-loss–based episode definition** adapted from Fearon et al. The main manuscript uses this operational definition consistently when referring to “episodes” detected from longitudinal BMI.
