from itertools import product
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification

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
effective_rank_percs = [0.1, 0.5]  # degree of colinearity
reg_noises = [0.1, 1, 10]            # variance of gaussian noise added


# generate simulated datasets
for random_seed, Nfeat, class_ratio, flip_y in product(random_seeds, Nfeats, class_ratios, flip_ys):
    cw = np.array([1, class_ratio])
    X, y = make_classification(
        n_samples=N, n_features=Nfeat,
        n_informative=round(Nfeat*feat_perc_informative),
        n_redundant=round(Nfeat*feat_perc_redundant),
        n_repeated=round(Nfeat*feat_perc_repeated),
        n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
        weights=cw/cw.sum(),
        flip_y=flip_y,
        random_state=random_seed)
    #X[:, :round(X.shape[1]*binary_ratio)] = (X[:, :round(X.shape[1]*binary_ratio)]>0).astype(int) # make binary
    df = pd.DataFrame(
        data=np.c_[X, y],
        columns=[f'x{i+1}' for i in range(X.shape[1])]+['y']
        )
    df.to_csv(os.path.join(simulation_data_dir, f'simulated_dataset_classfication_Nfeat{Nfeat}_classratio{class_ratio}_flipy{flip_y}_randomseed{random_seed}.csv'), index=False, float_format='%.1f')

    
for random_seed, Nfeat, effective_rank_perc, reg_noise in product(random_seeds, Nfeats, effective_rank_percs, reg_noises):
    X, y, coef = make_regression(
        n_samples=N, n_features=Nfeat,
        n_informative=round(Nfeat*feat_perc_informative),
        bias=10, coef=True,
        effective_rank=round(Nfeat*effective_rank_perc),
        tail_strength=0.5,
        noise=reg_noise,
        random_state=random_seed)
    X = X/np.mean(np.abs(X))*5  # sometimes magnitude of X is every small, make sure mean mag is 5
    #X[:, :round(X.shape[1]*binary_ratio)] = (X[:, :round(X.shape[1]*binary_ratio)]>0).astype(int) # make binary
    df = pd.DataFrame(
        data=np.c_[X, y],
        columns=[f'x{i+1}' for i in range(X.shape[1])]+['y']
        )
    df.to_csv(os.path.join(simulation_data_dir, f'simulated_dataset_regression_Nfeat{Nfeat}_effectiverank{effective_rank_perc}_noise{reg_noise}_randomseed{random_seed}.csv'), index=False, float_format='%.1f')

