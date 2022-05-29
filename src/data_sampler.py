import os
from get_data import read_params
import argparse
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

def over_sample_data(config_path):
    config = read_params(config_path)
    raw_data_path = config['load_data']['raw_dataset_csv']
    oversampled_data_path = config['sampled_data']['oversampled_dataset_csv']
    random_state = config['base']['random_state']
    
    
    df = pd.read_csv(
        raw_data_path, 
        sep=',', 
        encoding='utf-8'
        )
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    over_sampler = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = over_sampler.fit_resample(X,y)
    
    df_resampled = X_resampled
    df_resampled['TARGET'] = y_resampled
    df_resampled.to_csv(oversampled_data_path, sep=',', index=False, encoding='utf-8')
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml')
    parsed_args = args.parse_args()
    data = over_sample_data(parsed_args.config)