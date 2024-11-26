# cachexia
Episode Identification - BMI Trajectory

This project automates the identification of cachexia episodes using BMI data trajectories. It processes patient BMI data, applies smoothing techniques, and identifies episodes of significant BMI decrease, which are indicative of cachexia.

## Data Requirements
The following data formats:

- **BMI Data File (`bmi_data.csv`)**: Should contain at least the following fields:
  - `MRN` (Medical Record Number): Unique identifier for patients.
  - `datetime`: Date of the BMI measurement.
  - `BMI`: Body Mass Index measurement.

- **Metadata File (`metadata.csv`)**: Should contain at least the following fields:
  - `MRN` (Medical Record Number): Must correspond to the MRNs in the BMI data file.
  - `Tumor Diagnosis Date`: Date of diagnosis, used to calculate the days since diagnosis.

## Identification of cachectic episodes
### `load_data.py` - Data Processing
Clean and prepare BMI data through various functions:

- **`load_and_process_bmi_data(bmi_path, metadata_path)`**:
  - Standardizes data by days since diagnosis.
  - Filters out BMI outliers (BMI < 10 and BMI > 100).
  - Excludes patients with trajectories less than 180 days.

- **`smooth_bmi_ewma(df, smooth_col, alpha)`**:
  - Smooths data using Exponentially Weighted Moving Average (EWMA) where `alpha` is the smoothing factor. We used a default alpha of 0.2.

### `cachexia_identification.py` - Episode Identification
Detect cachexia episodes based on a defined threshold of BMI loss over time:

- **`identify_cachexia_episodes(df, time_col, bmi_col, recovery=True)`**:
  - Uses a sliding window to identify cachectic episodes.
  - Identifies start, onset, and end of each episode.

- **`identify_recovery_episodes(patient_data, merged_episodes_df, time_col, bmi_col)`**:
  - Identifies potential recoveries with a 5% increase in BMI if applicable.

- **`merge_episodes(df, start_col, end_col)`**:
  - Merges overlapping episodes to streamline episode data.
### `cac_qc.py`- Quality Control

- **`quality_control(episodes_file, output_path)`**:
  - Processes identified episodes to ensure each meets minimum duration (>=30 days), significance of weight loss (>5%) and start day after the tumor diagnosis (start_day>0).

### `main.py` - Execution Script

## Tumor genotype v.s. Cachexia
### `univariate_mutation.py` - Univariate competing risk model: CIF vs mutation
We used competing risk models to calculate the cumulative incidence of cachexia over time since tumor diagnosis, while properly accounting for death as a competing event. 
For each cancer type with more than 200 patients, we evaluated the statistical association between tumor genotypes and cachexia by modeling the incidence of cachexia episodes following tumor diagnosis as a function of oncogenic mutations with a mutation frequency greater than 5%.

  
