

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm

def load_and_process_bmi_data(bmi_path, metadata_path):
    metadata = pd.read_csv(metadata_path, parse_dates=['Tumor Diagnosis Date'])
    bmi_data = pd.read_csv(bmi_path)
    print("Initial data loaded.", metadata['MRN'].nunique())

    bmi_data.dropna(subset=['BMI'], inplace=True)
    bmi_data = bmi_data[(bmi_data['BMI'] >= 10) & (bmi_data['BMI'] <= 100)]
    
    bmi_data['datetime'] = pd.to_datetime(bmi_data['datetime'], errors='coerce')
    
    filtered_mrns = metadata['MRN'].unique()
    bmi_data = bmi_data[bmi_data['MRN'].isin(filtered_mrns)]
    
    print("Filtered [patient_identifier]:", bmi_data['MRN'].nunique())

    bmi_data = bmi_data.merge(metadata[['MRN', 'Tumor Diagnosis Date']], on='MRN', how='left')
    
    bmi_data['Days_Since_Diagnosis'] = (bmi_data['datetime'] - bmi_data['Tumor Diagnosis Date']).dt.days
    bmi_data = bmi_data[bmi_data['Days_Since_Diagnosis'] >= 0] #depends

    print("MRNs after > diagnosis:", bmi_data['MRN'].nunique())

    bmi_data = bmi_data.groupby('MRN').filter(lambda x: (x['datetime'].max() - x['datetime'].min()).days >= 180)
    print("load_data MRN", bmi_data['MRN'].nunique())

    return bmi_data

def smooth_bmi_ewma(df, smooth_col, alpha):
    """Applies exponential weighting to smooth BMI data."""
    df['smoothed_BMI'] = df[smooth_col].ewm(alpha=alpha, ignore_na=True).mean()
    return df
