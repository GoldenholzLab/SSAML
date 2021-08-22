import os
import shutil
from tqdm import tqdm
import pandas as pd


dataset_folder = '/data/interesting_side_projects/SSAML/datasets'
predict_folder = '/data/interesting_side_projects/SSAML/models_predictions'

# find the common file names that exist both in dataset and prediction
dataset_files = [x.replace('.csv','') for x in os.listdir(dataset_folder) if x.endswith('.csv')]
predict_files = [x.replace('_prediction_N100_modellogreg.csv','').replace('_prediction_N1000_modellogreg.csv','').replace('_prediction_N10000_modellogreg.csv','') for x in os.listdir(predict_folder) if x.endswith('.csv')]
common_files = sorted(set(dataset_files)&set(predict_files))

# move extra files to `notused_folder`
notused_folder = '/data/interesting_side_projects/SSAML/notused'
delete_dataset_files = [x for x in dataset_files if x not in common_files]
delete_predict_files = [x for x in predict_files if x not in common_files]
for ff in tqdm(delete_dataset_files):
    ff2 = os.path.join(dataset_folder, ff+'.csv')
    if os.path.exists(ff2):
        shutil.move(ff2, os.path.join(notused_folder, os.path.basename(ff2)))
for ff in tqdm(delete_predict_files):
    ff2 = os.path.join(predict_folder, ff+'_prediction_N100_modellogreg.csv')
    if os.path.exists(ff2):
        shutil.move(ff2, os.path.join(notused_folder, os.path.basename(ff2)))
    ff2 = os.path.join(predict_folder, ff+'_prediction_N1000_modellogreg.csv')
    if os.path.exists(ff2):
        shutil.move(ff2, os.path.join(notused_folder, os.path.basename(ff2)))
    ff2 = os.path.join(predict_folder, ff+'_prediction_N10000_modellogreg.csv')
    if os.path.exists(ff2):
        shutil.move(ff2, os.path.join(notused_folder, os.path.basename(ff2)))

# for file names with dataset and prediction both exist,
# add prediction as `p` column in the dataset and save
for ff in tqdm(common_files):
    df1 = pd.read_csv(os.path.join(dataset_folder, ff+'.csv'))
    if 'p' not in df1.columns:
        df2 = pd.read_csv(os.path.join(predict_folder, ff+'_prediction_N10000_modellogreg.csv'))
        assert len(df1)==len(df2)
        df3 = pd.concat([df1, df2], axis=1)
        df3.to_csv(os.path.join(dataset_folder, ff+'.csv'), index=False)

