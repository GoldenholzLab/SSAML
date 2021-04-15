import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm


simulation_data_dir = 'datasets'
df = pd.read_csv(os.path.join(simulation_data_dir, 'simulator_classification_dataset_list.csv'))

# generate simulated datasets
for i in range(len(df)):
    """
    runMode = int(sys.argv[1])
    peopleTF = int(sys.argv[2])==1
    iterNumber = int(sys.argv[3])
    maxPts = int(sys.argv[4])
    #confint options 0.95, 0.99, 0.999,...
    confint = float(sys.argv[5])
    survivalTF = int(sys.argv[6]) == 1
    big_file = sys.argv[7]
    mydir= sys.argv[8]
    """
    for n in [100,1000,10000]:
        input_file = os.path.join(os.getcwd(), 'temp.csv')
        output_dir = os.path.join(os.getcwd(), 'results')
        
        # generate different training sizes
        df_sim = pd.read_csv(df.Path.iloc[i])
        df_sim = df_sim[:n]  # because datset is shuffled, taking first N approximately keeps the class ratio
        df_sim.to_csv(input_file, index=False)
        
        runMode = 1
        peopleTF = 1
        iterNumber = 100
        maxPts = n
        confint = 0.95
        survivalTF = 0
        
        # run power.py
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        command = [
            'python3', '../power.py',
            runMode, peopleTF,
            iterNumber, maxPts,
            confint, survivalTF,
            input_file, output_dir]
        subprocess.check_call(command)
        os.remove(input_file)
    
