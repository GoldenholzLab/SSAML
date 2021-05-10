import os
import subprocess
import numpy as np
import pandas as pd


simulation_data_dir = 'datasets'
df = pd.read_csv(os.path.join(simulation_data_dir, 'simulator_classification_dataset_list.csv'))

model_type = 'logreg'

# generate simulated datasets
for i in range(len(df)):
    df_sim_all = pd.read_csv(df.Path.iloc[i])
    for n in [100,1000,10000]:
        print(i, n, df.Path.iloc[i])
        input_file = os.path.join(os.getcwd(), 'temp.csv')
        output_dir = os.path.join(os.getcwd(), 'results', os.path.basename(df.Path.iloc[i]).replace('simulated_dataset_', 'results_').replace('.csv',f'_N{n}'))
        
        # generate different training sizes
        df_sim = df_sim_all[:n]  # because datset is shuffled, taking first N approximately keeps the class ratio
        df_pred = pd.read_csv(os.path.join('models_predictions', os.path.basename(df.Path.iloc[i]).replace('.csv',f'_prediction_N{n}_model{model_type}.csv')))
        df_sim = pd.concat([df_sim, df_pred], axis=1)
        df_sim.to_csv(input_file, index=False)
        
        dataTYPE = '3'
        iterNumber = '100'
        maxPts = str(n)
        confint = '0.95'
        
        # run power.py
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        subprocess.check_call(['python3', '../power.py',
            '1', dataTYPE,
            iterNumber, maxPts, confint,
            input_file, output_dir])
        subprocess.check_call(['python3', '../power.py',
            '2', dataTYPE,
            iterNumber, maxPts, confint,
            input_file, output_dir])
        os.remove(input_file)
    
