from collections import defaultdict
from itertools import product
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# creates the output directory
simulation_data_dir = 'datasets'
if not os.path.exists(simulation_data_dir):
    os.mkdir(simulation_data_dir)

random_seeds = np.arange(2020, 2025) # repetition with different random seeds
N = 10000                            # big enough so that we can subsample to get different sizes
Nfeats = [10, 100, 500]              # number of features
feat_perc_informative = 0.6          # percent of informative features
#binary_ratio = 0.5

# classification specific
feat_perc_redundant = 0.1     # percent of redundant features
feat_perc_repeated = 0.1      # percent of repeated features
n_classes = 2                 # number of classes
n_clusters_per_class = 1      # number of clasters per class
class_ratios = [1, 10, 100]   # class 1:2 ratio
flip_ys = [0.01, 0.1, 0.2]    # probability of flipping class

# regression specific
#effective_rank_percs = [0.1, 0.5]  # degree of colinearity
#reg_noises = [0.1, 1, 10]            # variance of gaussian noise added


# generate simulated datasets
df_cls = defaultdict(list)
for random_seed, Nfeat, class_ratio, flip_y in product(random_seeds, Nfeats, class_ratios, flip_ys): # iterate over all combinations
    # print current parameters
    print(random_seed, Nfeat, class_ratio, flip_y)
    # define class weight
    cw = np.array([1, class_ratio])
    # call sklearn's make_classification function
    X, y = make_classification(
        n_samples=N, n_features=Nfeat,
        n_informative=round(Nfeat*feat_perc_informative),
        n_redundant=round(Nfeat*feat_perc_redundant),
        n_repeated=round(Nfeat*feat_perc_repeated),
        n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
        weights=cw/cw.sum(),
        flip_y=flip_y,
        random_state=random_seed)
    
    # add some monotonic nonlinearity to the features to make it harder
    X[:,:Nfeat//3] = X[:,:Nfeat//3]**3/10
    X[:,Nfeat//3:Nfeat//3*2] = 10*np.sign(X[:,Nfeat//3:Nfeat//3*2])*np.log1p(np.abs(X[:,Nfeat//3:Nfeat//3*2]))
    X[:,Nfeat//3*2:] = np.exp(X[:,Nfeat//3*2:])
    
    #X[:, :round(X.shape[1]*binary_ratio)] = (X[:, :round(X.shape[1]*binary_ratio)]>0).astype(int) # make binary
    # standardize
    std = X.std(axis=0)
    std[std<0.001] = 1
    X = X/std*5
    
    # convert into pandas dataframe and then save as csv
    df = pd.DataFrame(
        data=np.c_[X, y],
        columns=[f'x{i+1}' for i in range(X.shape[1])]+['event']
        )
    save_path = os.path.join(simulation_data_dir, f'simulated_dataset_classfication_Nfeat{Nfeat}_classratio{class_ratio}_flipy{flip_y}_randomseed{random_seed}.csv')
    df.to_csv(save_path, index=False, float_format='%.1f')
    df_cls['RandomSeed'].append(random_seed)
    df_cls['Nfeat'].append(Nfeat)
    df_cls['ClassRatio'].append(class_ratio)
    df_cls['FlipYProb'].append(flip_y)
    df_cls['Path'].append(save_path)

# save the overall dataset list
df_cls = pd.DataFrame(data=df_cls)
df_cls.to_csv(os.path.join(simulation_data_dir, 'simulator_classification_dataset_list.csv'), index=False)
