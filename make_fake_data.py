#!/usr/bin/python
# This code was designed to build a fake dataset to test out SSAML

import numpy as np
import os
import pandas as pd

# MAIN CODE

# CONSTANTS
COVAfile = 'COVA-FAKE.csv'
# First make fake COVA data
how_many = 2000
COVA = pd.DataFrame(columns=('actual','CoVAScore','Prob-none','Prob-Hosp','Prob-ICU-MV','Prob-dead'))
isTrue = np.random.randint(low=0,high=2,size=how_many) # random number 0 or 1
COVA['actual'] = isTrue*np.random.randint(low=0,high=4,size=how_many)    # a random integer from 0 to 3, nonzero if isTrue is 1
COVA['CoVAScore'] = np.random.random(how_many)*4    # a random floating number 0 to 4... we don't use this column anyway
goodPred = (np.abs(np.random.random(how_many)-isTrue)<.8)*1.0
noise = np.random.random(how_many)*.2
prob =  (goodPred==1)*(1-noise) + (goodPred==0)*noise
COVA['Prob-Hosp'] = prob*100
COVA['Prob-none'] = np.zeros(how_many)
COVA['Prob-ICU-MV'] = np.zeros(how_many)
COVA['Prob-dead'] = np.zeros(how_many)
COVA.to_csv(COVAfile,index=False,float_format='%.3f')
print('Done')