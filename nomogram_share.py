import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from sklearn.utils import resample
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.metrics import roc_curve

if __name__ == "__main__":
    # ----------------------------Load Data--------------------------------
    file_path = '...'
    results_dir = f'{file_path}/results/nomogram'
    cachexia_data = pd.read_csv(f'{file_path}/results/cachexia_data_5%.csv')
    metadata = pd.read_csv(f'{file_path}/data/raw_data/metadata_clin_1104.csv', header=0)
    bmi = pd.read_csv(f'{file_path}/data/processed_data/bmi_cleaned_upd_0913.csv', header=0)
    # replace the 'Tumor Diagnosis Date' in bmi with 'Tumor Diagnosis Date' in the metadata
    bmi = bmi.merge(metadata[['MRN', 'Tumor Diagnosis Date']], on='MRN', how='left')
    bmi['Days_Since_Diagnosis'] = (pd.to_datetime(bmi['datetime']) - pd.to_datetime(bmi['Tumor Diagnosis Date_y'])).dt.days
    bmi = bmi.dropna(subset=['Days_Since_Diagnosis'], axis=0)

    # ----------------------------Preprocess Data--------------------------------
    # calculate the age at diagnosis
    cachexia_data['Tumor Diagnosis Date'] = pd.to_datetime(cachexia_data['Tumor Diagnosis Date'])
    cachexia_data['PT_BIRTH_DTE'] = pd.to_datetime(cachexia_data['PT_BIRTH_DTE'])
    cachexia_data['age_at_diagnosis'] = (cachexia_data['Tumor Diagnosis Date'] - cachexia_data['PT_BIRTH_DTE']).dt.days / 3652.5  # scale by 10 years
    # drop na based on the features used in the nomogram model
    cachexia_data = cachexia_data.dropna(
        subset=[ 'STAGE_CDM_DERIVED_GRANULAR', 'ANCESTRY_LABEL'], axis=0)


    #----- add ECOG status data
    ecog = pd.read_csv(f'{file_path}/data/raw_data/data_timeline_ecog_kps.txt', sep='\t', header=0)
    ecog = ecog.sort_values(by=['PATIENT_ID', 'START_DATE'])
    ecog.reset_index(inplace=True )
    # get baseline ECOG status which is the closets to START_DATE 0
    ecog_baseline = ecog.groupby('PATIENT_ID', group_keys=False).apply(
        lambda x: x.loc[(x['START_DATE']).abs().idxmin()])
    ecog_baseline.rename(columns={'PATIENT_ID':'DMP_ID'}, inplace=True)
    # merge ecog_baseline with cachexia_data
    cachexia_data = cachexia_data.merge(ecog_baseline[['DMP_ID','ECOG_KPS']], on='DMP_ID', how='inner')

    # ----- add left right tumor info for colorectal cancer
    left_right_tumor = pd.read_csv(f'{file_path}/data/raw_data/CRC_Primary_Tumor_Location_01_24_25.csv')
    left_right_tumor = left_right_tumor.rename(columns={'PATIENT_ID': 'DMP_ID'})
    cachexia_data = cachexia_data.merge(left_right_tumor[['DMP_ID','Sidedness']], on='DMP_ID', how='left')

    # ----- add start BMI
    # get baseline start_BMI which is the closets to Days_Since_Diagnosis 0
    bmi_baseline = bmi.groupby('MRN', group_keys=False).apply(
        lambda x: x.loc[(x['Days_Since_Diagnosis']).abs().idxmin()])
    bmi_baseline.reset_index(inplace=True, drop=True)
    # merge bmi_baseline with cachexia_data
    cachexia_data = cachexia_data.merge(bmi_baseline[['MRN','BMI']], on='MRN', how='inner')
    cachexia_data.rename(columns={'BMI':'start_BMI'}, inplace=True)
    cachexia_data['start_BMI'] = cachexia_data['start_BMI']/5 # scale by 5
    # add a categorical variable for start_BMI (underweight, normal, overweight, obese)
    cachexia_data['start_BMI_cat'] = pd.cut(cachexia_data['start_BMI'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])

    # ----- add cachexia-associated lab values: albumin, ALK, HGB (only have lab data for 20,000 patients)
    time_bmilab = pd.read_csv(f'{file_path}/data/raw_data/time_bmilab.csv')
    time_bmilab = time_bmilab[['MRN', 'datetime', 'Albumin, Plasma', 'Alkaline Phosphatase (ALK), Plasma', 'HGB']]
    time_bmilab.rename(columns={'Albumin, Plasma': 'Albumin', 'Alkaline Phosphatase (ALK), Plasma':'ALK_plasma'}, inplace=True)
    time_bmilab = time_bmilab.dropna(subset=['datetime'], axis=0)
    time_bmilab['datetime'] = pd.to_datetime(time_bmilab['datetime'])
    time_bmilab = time_bmilab.merge(metadata[['MRN', 'Tumor Diagnosis Date']], on='MRN', how='inner').reset_index(drop=True)
    time_bmilab['Days_Since_Diagnosis'] = (time_bmilab['datetime'] - time_bmilab['Tumor Diagnosis Date']).dt.days
    for lab in ['Albumin', 'ALK_plasma', 'HGB']:
        df = time_bmilab.copy()
        df = df.dropna(subset=[lab], axis=0)
        baseline = df.groupby('MRN', group_keys=False).apply(
            lambda x: x.loc[(x['Days_Since_Diagnosis']).abs().idxmin()])
        baseline.reset_index(inplace=True, drop=True)
        # merge bmi_baseline with cachexia_data
        cachexia_data = cachexia_data.merge(baseline[['MRN', lab]], on='MRN', how='inner')



    # -----------------------------Build nomogram--------------------------------
    cancer_type = 'Lung Adenocarcinoma'
    df = cachexia_data[cachexia_data['CANCER_TYPE_DETAILED'] == cancer_type]
    # CRC
    #variables_5 = [ "ANCESTRY_LABEL", 'MSI_TYPE', 'start_BMI',
    #                                 "STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS", "Sidedness", 'SMAD4'] # 365,730,1095 days
    #variables_10 = [ "ANCESTRY_LABEL", 'start_BMI','age_at_diagnosis',
    #                                 "STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS"] # 365, 729, 1095 days
    #variables_15 = ["STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS", 'start_BMI','age_at_diagnosis'] # 365, 730, 1096 days
    # LUAD
    variables_5 = [ "ANCESTRY_LABEL",'start_BMI',
                                    "STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS", 'TP53', 'STK11']
    variables_10 = [ "ANCESTRY_LABEL", 'start_BMI','age_at_diagnosis',
                                     "STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS",'TP53', 'STK11']
    variables_15 = ["STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS", 'start_BMI','age_at_diagnosis']
    df = df.loc[df['time_to_cachexia']>0]
    df = df.dropna(subset=['MSI_TYPE', "Sidedness"], axis=0)

    # Build the formula
    formula = " + ".join(variables_15)
    # fit the model
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time_to_cachexia', event_col='cachexia_event',
            formula=formula)
    # save the nomogram model
    with open(f'{results_dir}/cachexia_nomogram_model_LUAD_5%.pkl', 'wb') as f:
        pickle.dump(cph, f)

    # bootstrapping and Validation
    n_bootstrap = 200
    c_indices = []

    for i in range(n_bootstrap):
        # Resample dataset
        df_resampled = resample(df, replace=True, n_samples=len(df), random_state=i)
        df_resampled.reset_index(drop=True, inplace=True)
        df_resampled['partial_hazard'] = cph.predict_partial_hazard(df_resampled)
        survival_probability = cph.predict_survival_function(df_resampled)

        # Compute C-index
        c_index = concordance_index(df_resampled['time_to_cachexia'],
                                    -cph.predict_partial_hazard(df_resampled),
                                    df_resampled['cachexia_event'])
        c_indices.append(c_index)


    c_index_mean = np.mean(c_indices)  # 0.68
    c_index_ci = (np.percentile(c_indices, 2.5), np.percentile(c_indices, 97.5))
    print(c_index_mean, c_index_ci)



    # --------------- Validation 1: Calibration Plot ---------------
    # -----------------------------50/50 split
    # ----------------------------- Step 1: Randomly Split Dataset (50% Train, 50% Test) -----------------------------
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)

    # ----------------------------- Step 2: Fit Cox Model on Training Data -----------------------------
    cph_split = CoxPHFitter()
    cph_split.fit(df_train, duration_col='time_to_cachexia', event_col='cachexia_event', formula=formula)

    # ----------------------------- Step 3: Predict Cachexia Probability for Test Set -----------------------------
    # Get available time points in the survival function
    available_times = cph_split.predict_survival_function(df_test).index.tolist()

    # Define required time points (check if available)
    time_points = [t for t in [365, 730, 1096] if t in available_times]

    predicted_risks = {}

    for t in time_points:
        df_test[f'predicted_survival_{t}d'] = cph_split.predict_survival_function(df_test).loc[t]
        df_test[f'predicted_cachexia_{t}d'] = 1 - df_test[f'predicted_survival_{t}d']
        predicted_risks[t] = df_test[f'predicted_cachexia_{t}d']

    # ----------------------------- Step 4: Bin Test Patients Using Quantiles -----------------------------
    df_test['risk_bin'] = pd.qcut(df_test['predicted_cachexia_365d'], q=4, labels=False)

    # ----------------------------- Step 5: Compute Observed Cachexia Risk per Bin with Bootstrapping -----------------------------
    kmf = KaplanMeierFitter()
    observed_cachexia = {t: [] for t in time_points}
    ci_lower, ci_upper = {t: [] for t in time_points}, {t: [] for t in time_points}

    n_bootstrap = 200  # Number of bootstrap resamples

    for group in sorted(df_test['risk_bin'].unique()):
        group_data = df_test[df_test['risk_bin'] == group]

        for t in time_points:
            kmf.fit(group_data['time_to_cachexia'], event_observed=group_data['cachexia_event'])
            observed_risk = 1 - kmf.predict(t)  # Convert survival probability to cachexia probability
            observed_cachexia[t].append(observed_risk)

            # Bootstrap to get confidence intervals
            bootstrap_estimates = []
            for _ in range(n_bootstrap):
                resampled_data = resample(group_data, replace=True, n_samples=len(group_data), random_state=_)
                kmf.fit(resampled_data['time_to_cachexia'], event_observed=resampled_data['cachexia_event'])
                bootstrap_estimates.append(1 - kmf.predict(t))

            # Compute 95% CI
            ci_lower[t].append(np.percentile(bootstrap_estimates, 2.5))
            ci_upper[t].append(np.percentile(bootstrap_estimates, 97.5))

    # ----------------------------- Step 6: Create Calibration Data -----------------------------
    df_calibration = {}

    for t in time_points:
        df_calibration[t] = pd.DataFrame({
            "Predicted Cachexia Risk": df_test.groupby('risk_bin')[f'predicted_cachexia_{t}d'].mean(),
            "Observed Cachexia Risk": observed_cachexia[t],
            "CI Lower": ci_lower[t],
            "CI Upper": ci_upper[t]
        })

    # ----------------------------- Step 7: Plot Calibration Curve with Confidence Intervals -----------------------------
    plt.figure(figsize=(7, 7))

    # Define markers for each time point
    markers = {365: 'o', 730: 's', 1096: '^'}
    colors = {365: 'blue', 730: 'orange', 1096: 'green'}

    for t in time_points:
        plt.errorbar(df_calibration[t]["Predicted Cachexia Risk"], df_calibration[t]["Observed Cachexia Risk"],
                     yerr=[df_calibration[t]["Observed Cachexia Risk"] - df_calibration[t]["CI Lower"],
                           df_calibration[t]["CI Upper"] - df_calibration[t]["Observed Cachexia Risk"]],
                     fmt=markers[t], label=f"{round(t/365)}-year cachexia incidence", color=colors[t], capsize=3)

    plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration", color="black")
    plt.xlabel("Nomogram-Predicted Probability of Cachexia")
    plt.ylabel("Cumulative Incidence of Cachexia")
    plt.title("Calibration Plot on Test Cohort (50/50 Split, Bootstrapped CIs)")
    plt.legend()
    plt.savefig(f'{results_dir}/calibration_half_split_validation.pdf')
    plt.close()



    # ----------------------------- Validation 2: AUC -----------------------------#
    # ----------------------------- Step 1: Randomly Split Dataset (50% Train, 50% Test) -----------------------------
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)

    # ----------------------------- Step 2: Fit Cox Model on Training Data -----------------------------
    cph_split = CoxPHFitter()
    cph_split.fit(df_train, duration_col='time_to_cachexia', event_col='cachexia_event', formula=formula)

    # ----------------------------- Step 3: Predict Cachexia Risk for Test Set -----------------------------
    # Get available time points in the survival function
    available_times = cph_split.predict_survival_function(df_test).index.tolist()

    # Define required time points (check if available)
    time_points = [t for t in [365, 730, 1096] if t in available_times]

    # Compute predicted survival and cachexia probability
    predicted_risks = {}
    for t in time_points:
        df_test[f'predicted_survival_{t}d'] = cph_split.predict_survival_function(df_test).loc[t]
        df_test[f'predicted_cachexia_{t}d'] = 1 - df_test[f'predicted_survival_{t}d']
        predicted_risks[t] = df_test[f'predicted_cachexia_{t}d']

    # ----------------------------- Step 4: Compute Time-Dependent AUC -----------------------------
    # Convert test set to structured array format for sksurv
    survival_train = Surv.from_dataframe(event="cachexia_event", time="time_to_cachexia", data=df_train)
    survival_test = Surv.from_dataframe(event="cachexia_event", time="time_to_cachexia", data=df_test)

    # Compute AUC at each time point
    auc_values, fpr_tpr_dict = {}, {}

    plt.figure(figsize=(7, 7))

    # Define colors for each time point
    colors = {365: 'red', 730: 'green', 1096: 'blue'}

    for t in time_points:
        auc_values[t], _ = cumulative_dynamic_auc(
            survival_train, survival_test, df_test[f'predicted_cachexia_{t}d'], [t]
        )

        # Compute time-specific event labels
        df_test[f'cachexia_event_{t}d'] = (df_test['cachexia_event'] & (df_test['time_to_cachexia'] <= t)).astype(int)

        # Compute ROC curve using time-censored labels
        fpr, tpr, _ = roc_curve(df_test[f'cachexia_event_{t}d'], df_test[f'predicted_cachexia_{t}d'])

        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[t], label=f"{round(t/365)}-Year AUC: {auc_values[t][0]:.3f}")

    # ----------------------------- Step 5: Plot ROC Curves -----------------------------
    plt.plot([0, 1], [0, 1], "--", color="black", label="Random Chance")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Time-Dependent ROC Curves for Cachexia Prediction")
    plt.legend()
    plt.savefig(f'{results_dir}/auc_half_split_validation.pdf')
    plt.close()

    # Print AUC values
    for t in time_points:
        print(f"AUC at {round(t/365)} year(s): {auc_values[t][0]:.3f}")