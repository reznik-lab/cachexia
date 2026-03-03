

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm



def identify_cachexia_episodes(df, time_col, bmi_col, recovery=True):
    '''
    Detect cachexia episodes, including start, onset, end, and recovery information.
    '''
    results = []
    
    df.reset_index(drop=True, inplace=True) 
    
    for i in range(df.shape[0]):
        t0 = df[time_col][i]  
        b0 = df[bmi_col][i]  
        b0_raw = df['smoothed_BMI'][i] 
        current_date = df['datetime'][i] 
        current_bmi = df['BMI'][i]  
        df_window = df[(df[time_col] > t0) & (df[time_col] <= t0 + 180)]
        
        df_filtered = df_window[df_window[bmi_col] < b0 + np.log(0.95)]
        
        if not df_filtered.empty:
            # Onset: First day when BMI falls below the threshold
            onset_day = df_filtered[time_col].min()
            onset_date = df_filtered.loc[df_filtered[time_col] == onset_day, 'datetime'].values[0] if not df_filtered.loc[df_filtered[time_col] == onset_day].empty else None
            #onset_bmi = df_filtered.loc[df_filtered[time_col] == onset_day, bmi_col].values[0] if not df_filtered.loc[df_filtered[time_col] == onset_day].empty else None
            onset_bmi= df_filtered.loc[df_filtered[time_col] == onset_day, 'smoothed_BMI'].values[0] if not df_filtered.loc[df_filtered[time_col] == onset_day].empty else None

            # End of the episode: Last day where BMI is still below the threshold
            max_t = df_filtered[time_col].max()
            end_date = df_filtered.loc[df_filtered[time_col] == max_t, 'datetime'].values[0] if not df_filtered.loc[df_filtered[time_col] == max_t].empty else None
            #end_bmi = df_filtered.loc[df_filtered[time_col] == max_t, bmi_col].values[0] if not df_filtered.loc[df_filtered[time_col] == max_t].empty else None
            end_bmi = df_filtered.loc[df_filtered[time_col] == max_t, 'smoothed_BMI'].values[0] if not df_filtered.loc[df_filtered[time_col] == max_t].empty else None
 
            # Append episode information
            results.append({
                'start_day': t0,
                'start_date': current_date,
                'start_bmi': current_bmi,
                'onset_day': onset_day,
                'onset_date': onset_date,
                'onset_bmi': onset_bmi,
                'end_day': max_t,
                'end_date': end_date,
                'end_bmi': end_bmi
            })
    
    # If no results (i.e., no episodes) found, append a row with 'None' values for this MRN
    if len(results) == 0:
        # You can choose what columns to include if no episodes are detected
        results.append({
            'start_day': None,
            'start_date': None,
            'start_bmi': None,
            'onset_day': None,
            'onset_date': None,
            'onset_bmi': None,
            'end_day': None,
            'end_date': None,
            'end_bmi': None
        })

    # Merge overlapping episodes
    merged_episodes_df = merge_episodes(pd.DataFrame(results), start_col='start_day', end_col='end_day')
    
    # Identify recovery episodes if specified
    if recovery:
        merged_episodes_df = identify_recovery_episodes(df, merged_episodes_df, time_col, bmi_col)

    return merged_episodes_df

def identify_recovery_episodes(patient_data, merged_episodes_df, time_col, bmi_col):
    # Initialize new columns for recovery day, date, and BMI
    merged_episodes_df['recovery_day'] = None
    merged_episodes_df['recovery_date'] = None
    merged_episodes_df['recovery_bmi'] = None
    
    merged_episodes_df['recovery_smoothed_bmi'] = None 
    if not merged_episodes_df['start_day'].isna().all():
        for i in range(merged_episodes_df.shape[0]):
            t0 = merged_episodes_df['end_day'][i]
            if i < merged_episodes_df.shape[0]-1:
                t1 = merged_episodes_df['start_day'][i+1]
            else:
                t1 = patient_data[time_col].max()
                
            # Get the BMI at the end of the episode
            end_index = patient_data.loc[patient_data[time_col] == t0].index[0]
            b0 = patient_data[bmi_col][end_index]

            # Detect recovery episodes: BMI greater than 5% from the lowest value at the end of the episode
            recovery_data = patient_data[(patient_data[time_col] >= t0) & (patient_data[time_col] <= t1)
                                         & (patient_data[bmi_col] > b0 + np.log(1.05))]
            
            if not recovery_data.empty:
                recovery_day = recovery_data[time_col].min()
                recovery_date = recovery_data.loc[recovery_data[time_col] == recovery_day, 'datetime'].values[0]
                recovery_bmi = recovery_data.loc[recovery_data[time_col] == recovery_day, bmi_col].values[0]
                recovery_smoothed_bmi = recovery_data.loc[recovery_data[time_col] == recovery_day, 'smoothed_BMI'].values[0]
                
                
                # Assign recovery information to the merged episodes dataframe
                merged_episodes_df.at[i, 'recovery_day'] = recovery_day
                merged_episodes_df.at[i, 'recovery_date'] = recovery_date
                merged_episodes_df.at[i, 'recovery_bmi'] = recovery_bmi
                merged_episodes_df.at[i, 'recovery_smoothed_bmi'] = recovery_smoothed_bmi  # Save smoothed BMI
                
    return merged_episodes_df



def merge_episodes(df, start_col, end_col):
    '''
    Merges overlapping cachexia episodes, keeping start, onset, end, and associated dates/BMI.
    '''
    merged_episodes = []
    current_episode = None

    for i in range(df.shape[0]):
        if current_episode is None:
            # Initialize the first episode
            current_episode = df.iloc[i].to_dict()
        elif df[start_col][i] <= current_episode['end_day']:
            # Merge episodes if overlapping
            current_episode['end_day'] = max(current_episode['end_day'], df[end_col][i])
            current_episode['end_date'] = df.loc[i, 'end_date']
            current_episode['end_bmi'] = df.loc[i, 'end_bmi']
        else:
            # No overlap, append the previous episode and start a new one
            merged_episodes.append(current_episode)
            current_episode = df.iloc[i].to_dict()

    # Add the last episode
    if current_episode is not None:
        merged_episodes.append(current_episode)

    return pd.DataFrame(merged_episodes)
