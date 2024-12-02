import os
import sys

# Set the working directory to where your scripts are
script_directory = '/Users/castilv/Documents/Cachexia/cac_data/scripts/cac_identification'
os.chdir(script_directory)

# Ensure the directory is added to Python's search path
sys.path.append(script_directory)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter

from load_data import load_and_process_bmi_data, smooth_bmi_ewma
from cachexia_identification import identify_cachexia_episodes, identify_recovery_episodes, merge_episodes
from cac_qc import quality_control
from competing_risk_genotype_test import define_competing_events, univariate_mutation_test, multivariate_mutation_test

bmi_data_path = '/Users/castilv/Documents/Cachexia/cac_data/bmi_upd.csv'
metadata_path = '/Users/castilv/Documents/Cachexia/cac_data/clean_metadata/metadata_clin_0930.csv'
mutation_path = '/Users/xiea1/cachexia/data/raw_data/mutation.csv'
output_path = '/Users/castilv/Documents/Cachexia/cac_data/output/'

def main():
    # Preprocess
    bmi_data = load_and_process_bmi_data(bmi_data_path, metadata_path)
    bmi_data = bmi_data.groupby('MRN').apply(lambda x: smooth_bmi_ewma(x, 'BMI', 0.2)).reset_index(drop=True)
    bmi_data['log_smoothed_BMI'] = np.log(bmi_data['smoothed_BMI'])
    bmi_smoothed_file = output_path + 'bmi_smooth.csv'
    bmi_data.to_csv(bmi_smoothed_file, index=False)
    print(f"Smoothed BMI data saved to: {bmi_smoothed_file}")
    
    # Identify cachexia episodes
    df_episodes_all = pd.DataFrame()
    for mrn in tqdm(bmi_data['MRN'].unique(), desc="Processing MRNs"):
        df = bmi_data[bmi_data['MRN'] == mrn]
        episodes = identify_cachexia_episodes(df, 'Days_Since_Diagnosis', 'log_smoothed_BMI')
        episodes['MRN'] = mrn  # Ensure MRN is carried forward
        df_episodes_all = pd.concat([df_episodes_all, episodes], ignore_index=True)

    episodes_file = output_path + 'processed_episodes.csv'
    df_episodes_all.to_csv(episodes_file, index=False)
    print("Analysis complete. Processed episodes saved to:", episodes_file)

    valid_episodes = quality_control(episodes_file, output_path)
    
    # Load Mutation and test with mutations
    mutation = pd.read_csv(mutation_path, header=0)
    # convert mutation data to binary, any value larger than 1 is 1
    for gene in mutation.columns[1:]:
        mutation.loc[mutation[gene] > 1, gene] = 1

    cachexia_data = define_competing_events(df_episodes_all, valid_episodes, mutation, metadata, output_path)
    cox_df_uni = univariate_mutation_test(cachexia_data, mutation, output_path)
    cox_df_multi = multivariate_mutation_test(cachexia_data, mutation, output_path)

if __name__ == "__main__":
    main()
    
