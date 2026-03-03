

import pandas as pd
import numpy as np

def quality_control(episodes_file, output_path):
    
    df_episodes_all = pd.read_csv(episodes_file)
    
    valid_episodes = df_episodes_all.dropna(subset=['start_day', 'end_day'])
    
    episode_counts = valid_episodes.groupby('MRN').size().reset_index(name='episode_count')
    all_patients = pd.DataFrame(df_episodes_all['MRN'].unique(), columns=['MRN'])
    episode_summary = pd.merge(all_patients, episode_counts, on='MRN', how='left').fillna(0)
    episode_summary['episode_count'] = episode_summary['episode_count'].astype(int)
    
    #Filter/ QC for duration, weight loss and episode start day (It needs to be after the tumor diagnosis.)
    valid_episodes['episode_duration'] = (valid_episodes['end_day'] - valid_episodes['start_day']).astype(int)
    valid_episodes = valid_episodes[valid_episodes['episode_duration'] >= 30]
    valid_episodes['weight_loss'] = (valid_episodes['start_bmi'] - valid_episodes['end_bmi']) / valid_episodes['start_bmi'] * 100
    valid_episodes = valid_episodes[valid_episodes['weight_loss'] > 5]
    valid_episodes = valid_episodes[valid_episodes['start_day'] > 0]

    valid_episodes_file = f'{output_path}/valid_cachexia_episodes_filtered.csv'
    episode_summary_file = f'{output_path}/episode_summary.csv'
    
    valid_episodes.to_csv(valid_episodes_file, index=False)
    episode_summary.to_csv(episode_summary_file, index=False)
    
    print(f"Total valid episodes after filtering: {valid_episodes.shape[0]}")
    print("Analysis complete. Files saved to:", output_path)
    return valid_episodes

if __name__ == "__main__":
    episodes_file = '/path/to/episodes_data.csv'
    output_path = '/path/to/output_directory'
    valid_episodes = quality_control(episodes_file, output_path)
