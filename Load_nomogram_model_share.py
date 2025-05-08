import pickle
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

if __name__ == "__main__":
    # -----------------------------Load the LUAD nomogram model and new patient data--------------------------------
    model_path = 'cachexia_nomogram_model_LUAD_5%.pkl'  # Path to the saved model
    with open(model_path, 'rb') as f:
        cph = pickle.load(f)
    patient_data = pd.read_csv('../data/patient_data.csv')  # Replace with actual path to your patient data
    # Ensure the patient data are formatted as each row is a patient, include columns with the same name in the model

    # -----------------------------Option 1: use lifelines function--------------------------------
    survival_probability = cph.predict_survival_function(patient_data)
    cachexia_probability = 1 - survival_probability
    # Here each column of cachexia_probability represents a patient, and each row represents the probability of cachexia at a specific day after diagnosis.

    # -----------------------------Option 2: manually extract predictive formulas--------------------------------
    # Extract the coefficients (β_i) from your fitted model:
    coef = cph.params_
    # Obtain Baseline Survival at Specific Times (365, 730, 1095 days)
    baseline_survival = cph.baseline_survival_
    S_0_1yr = baseline_survival.loc[365].values[0]
    S_0_2yr = baseline_survival.loc[731].values[0]
    S_0_3yr = baseline_survival.loc[1096].values[0]

    # Normalize the patient data by subtracting the mean of training data
    X_centered = patient_data[coef.index] - cph._norm_mean[coef.index]
    linear_predictor = np.dot(X_centered, coef.values)

    # Calculate risks at specific days
    risk_1yr = 1 - (S_0_1yr ** np.exp(linear_predictor))
    risk_2yr = 1 - (S_0_2yr ** np.exp(linear_predictor))
    risk_3yr = 1 - (S_0_3yr ** np.exp(linear_predictor))

    # Combine results into a single DataFrame for clarity
    risk_df = patient_data.copy()
    risk_df['risk_365'] = risk_1yr
    risk_df['risk_731'] = risk_2yr
    risk_df['risk_1096'] = risk_3yr

    # Display results
    print(risk_df[['risk_365', 'risk_731', 'risk_1096']])
