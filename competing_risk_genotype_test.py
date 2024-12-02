import pandas as pd
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter


def define_competing_events(df_episodes_all, valid_episodes, mutation, metadata, results_dir):
    '''
    Define the cachexia event and time_to_cachexia (time from the tumor diagnosis date to the first cachexia episode),
    Then merge with metadata to get os_days and os_event (for the competing risk model)
    '''
    cachexia_data = valid_episodes.groupby('MRN').apply(
        lambda x: x.loc[x['start_day'].idxmin()]).reset_index(drop=True)
    # fill in the missing values, for mrns unique in df_episodes_all, but not in cachexia_data, add them to cachexia_data
    missing_mrns = set(df_episodes_all['MRN']) - set(cachexia_data['MRN'])
    # create a new dataframe with missing mrns, column 'MRN'
    missing_data = pd.DataFrame(list(missing_mrns), columns=['MRN'])
    cachexia_data = pd.concat([cachexia_data, missing_data], axis=0)
    cachexia_data = cachexia_data[['start_day', 'MRN']]
    cachexia_data = cachexia_data.rename(columns={'start_day': 'time_to_cachexia'})
    cachexia_data['cachexia_event'] = 0
    cachexia_data.loc[cachexia_data['MRN'].isin(valid_episodes['MRN']), 'cachexia_event'] = 1

    # merge cachexia_data with metadata to get os_days and os_event
    cachexia_data = cachexia_data.merge(metadata, on='MRN', how='left')
    cachexia_data = cachexia_data.dropna(subset=['os_days', 'os_event'])
    # if cachexia_event is 0, set it to os_days
    cachexia_data.loc[cachexia_data['cachexia_event'] == 0, 'time_to_cachexia'] = cachexia_data['os_days']

    # define multi-state events
    # create a new event column, if os_event is 2, cachexia_event is 1, neither is 0
    cachexia_data['event'] = 0
    cachexia_data.loc[cachexia_data['os_event'] == 1, 'event'] = 2
    cachexia_data.loc[cachexia_data['cachexia_event'] == 1, 'event'] = 1
    # create a new time column, if cachexia_event is 1, time is time_to_cachexia, otherwise os_months
    cachexia_data['time'] = cachexia_data['os_days']
    cachexia_data.loc[cachexia_data['cachexia_event'] == 1, 'time'] = cachexia_data['time_to_cachexia']

    # merge with mutation data
    cachexia_data = cachexia_data.merge(mutation, on='DMP_ID', how='left')
    cachexia_data = cachexia_data[cachexia_data['AR'].notna()]
    # cachexia_data.to_csv(f'{file_path}/data/processed_data/cachexia_data_survival_mutation_0919_after_diag.csv', index=False)
    cachexia_data.to_csv(f'{results_dir}/cachexia_data_survival_mutation_0919_30days_15%.csv',
                         index=False)
    return cachexia_data
def univariate_mutation_test(cachexia_data, mutation, results_dir):
    '''
    For 341 oncogenic genes in each detailed_cancer_type:
        time_to_cachexia ~ mutation
    '''
    # -----------------------------Cox Proportional Hazard Model: cachexia--------------------------------
    cancer_counts = cachexia_data['CANCER_TYPE_DETAILED'].value_counts().reset_index()
    all_genes = mutation.columns[1:]
    cox_df = pd.DataFrame(
        {'coef': [], 'exp(coef)': [], 'se(coef)': [], 'coef lower 95%': [], 'coef upper 95%': [],
         'exp(coef) lower 95%': [], 'exp(coef) upper 95%': [], 'cmp to': [], 'z': [], 'p': [], '-log2(p)': []})
    for cancer_type in cancer_counts.loc[cancer_counts['count'] > 200, 'CANCER_TYPE_DETAILED']:
        df = cachexia_data[cachexia_data['CANCER_TYPE_DETAILED'] == cancer_type]
        # find the genes that are mutated in at least 5% of the patients
        mutation_rate = df[all_genes].sum() / df.shape[0]
        filtered_genes = mutation_rate[mutation_rate > 0.05].index.to_list()
        if len(filtered_genes) == 0:
            continue
        else:
            for gene in filtered_genes:
                data = df[df[gene].notna()]
                # fit the model
                cph = CoxPHFitter()
                cph.fit(data, duration_col='time_to_cachexia', event_col='cachexia_event',
                        formula=f'{gene}')
                cox_df.loc[f'{cancer_type}_{gene}'] = cph.summary.iloc[0, :]
                cox_df.loc[f'{cancer_type}_{gene}', 'detailed_cancer_type'] = cancer_type
                cox_df.loc[f'{cancer_type}_{gene}', 'mutation'] = gene
    cox_df['p_adj'] = multipletests(cox_df['p'], method='fdr_bh')[1]
    cox_df.to_csv(f'{results_dir}/mutation_cox_cachexia_results_univariate.csv')
    return cox_df

def multivariate_mutation_test(cachexia_data, mutation, results_dir):
    '''
    For 341 oncogenic genes in each detailed_cancer_type:
        time_to_cachexia ~ mutation + age_at_diagnosis + GENDER + genetic ancestry + SAMPLE_TYPE + CVR_TMB_SCORE + pathologic stage
    '''
    # --------------Preprocessing for the multivariate cox model--------------
    # calculate the age at diagnosis
    cachexia_data['Tumor Diagnosis Date'] = pd.to_datetime(cachexia_data['Tumor Diagnosis Date'])
    cachexia_data['PT_BIRTH_DTE'] = pd.to_datetime(cachexia_data['PT_BIRTH_DTE'])
    cachexia_data['age_at_diagnosis'] = (cachexia_data['Tumor Diagnosis Date'] - cachexia_data[
        'PT_BIRTH_DTE']).dt.days / 365.25
    # drop na
    cachexia_data = cachexia_data.dropna(
        subset=['age_at_diagnosis', 'GENDER', 'ANCESTRY_LABEL', 'SAMPLE_TYPE', 'CVR_TMB_SCORE',
                'STAGE_CDM_DERIVED_GRANULAR'], axis=0)

    # -----------------------------Multivariate Cox Model: cachexia--------------------------------
    cancer_counts = cachexia_data['CANCER_TYPE_DETAILED'].value_counts().reset_index()
    all_genes = mutation.columns[1:]
    cox_df = pd.DataFrame(
        {'coef': [], 'exp(coef)': [], 'se(coef)': [], 'coef lower 95%': [], 'coef upper 95%': [],
         'exp(coef) lower 95%': [], 'exp(coef) upper 95%': [], 'cmp to': [], 'z': [], 'p': [], '-log2(p)': []})
    for cancer_type in cancer_counts.loc[cancer_counts['count'] > 200, 'CANCER_TYPE_DETAILED']:
        df = cachexia_data[cachexia_data['CANCER_TYPE_DETAILED'] == cancer_type]
        # find the genes that are mutated in at least 5% of the patients
        mutation_rate = df[all_genes].sum() / df.shape[0]
        filtered_genes = mutation_rate[mutation_rate > 0.05].index.to_list()
        if len(filtered_genes) == 0:
            continue
        else:
            for gene in filtered_genes:
                data = df[df[gene].notna()]
                variables = [f"{gene}", "age_at_diagnosis", "GENDER", "ANCESTRY_LABEL", "SAMPLE_TYPE", "CVR_TMB_SCORE",
                             "STAGE_CDM_DERIVED_GRANULAR"]
                # variables = [var for var in variables if data[var].value_counts(normalize=True).iloc[0] < 0.99]

                # Build the formula
                formula = " + ".join(variables)
                # fit the model
                cph = CoxPHFitter()
                cph.fit(data, duration_col='time_to_cachexia', event_col='cachexia_event',
                        formula=formula)
                cox_df.loc[f'{cancer_type}_{gene}'] = cph.summary.iloc[0, :]
                cox_df.loc[f'{cancer_type}_{gene}', 'detailed_cancer_type'] = cancer_type
                cox_df.loc[f'{cancer_type}_{gene}', 'mutation'] = gene

    cox_df['p_adj'] = multipletests(cox_df['p'], method='fdr_bh')[1]
    cox_df.to_csv(f'{results_dir}/mutation_cox_cachexia_results_multivariate.csv')
    return cox_df



