import numpy as np
from scipy.stats import cauchy
from scipy.special import logit, expit
import pandas as pd


datasets = ['bai_mortality', 'cova', 'st']
outlier_percents = [0,5,10]

np.random.seed(2023)

for dataset in datasets:
    df = pd.read_csv(f'../sample_data_{dataset}.csv')
    if dataset=='bai_mortality':
        val_col = 'z'
        transform = lambda x:x  # identity
        inv_transform = lambda x:x  # identity
    else:
        val_col = 'p'
        transform = lambda x:logit(np.clip(x,1e-5,1-1e-5))
        inv_transform = expit
    for outlier_percent in outlier_percents:
        df2 = df.copy()
        df2.loc[:,val_col] = transform(df2[val_col])
        """
        if outlier_scale!=1:
            # estimate parameter for Cauchy distribution
            mu_ = np.median(df2[val_col])
            q1, q3 = np.percentile(df2[val_col], (25,75))
            lambda_ = (q3-q1)/2
            cauchy.fit(df2[val_col])
            r = cauchy.rvs(size=1000)
        """
        outlier_ids = np.random.choice(len(df2), int(round(len(df2)*outlier_percent/100)), replace=False)
        df2.loc[outlier_ids, val_col] *= 10
        df2.loc[:,val_col] = inv_transform(df2[val_col])

        df2.to_csv(f'sample_data_{dataset}_outlier_perc{outlier_percent}.csv', index=False)

