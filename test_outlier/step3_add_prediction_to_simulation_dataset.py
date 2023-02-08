import os
import pandas as pd

df = pd.read_csv('datasets/simulator_classification_dataset_list.csv')

for i in range(len(df)):
    df1 = pd.read_csv(df.Path.iloc[i])
    if 'p' not in df1.columns:
        df2 = pd.read_csv(os.path.join('models_predictions', os.path.basename(df.Path.iloc[i]).replace('.csv', '_prediction_N1000_modellogreg.csv')))
        df1['ID'] = list(range(len(df1)))
        df1['event'] = df1.event.astype(int)
        df3 = pd.concat([df1[['ID', 'event']], df2], axis=1)
        df3.to_csv(df.Path.iloc[i], index=False)
