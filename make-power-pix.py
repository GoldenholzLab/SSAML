#!/usr/bin/python
# This script will produce the pictures from our study based on the output of SSAML

# First load up the libraries needed
# this for drawing headless
import matplotlib

#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
import time

# CONSTANTS
# put whatever your local directory is that has your files from SSAML
mydir = '/Users/danisized/iCloud/Latest//power-calculator-stat/Zfiles'
# Note, the numLIST numbers here are hard coded for the number of patients/events we tested. 
# change to whatever you like here
numLIST_bai = [500,1000,1500,2000]
numLIST_cov = [125, 150, 175, 200]
numLIST_st = [20,40,60,80]

# FUNCTION DEFINITIONS
def getZING(prefixN,middleOne,numLIST):
  # load up the ZING files and compose a pandas dataframe from it
  print('Loading %s...' % prefixN)
  
  for howmany in numLIST:
    fn = prefixN + str(howmany).zfill(4) + '.csv'
    dat = pd.read_csv(fn,sep=',',header=None)
    dat.columns =['Slope',middleOne,'CIL']
    dat['N'] = dat.Slope*0 + howmany
    if howmany == numLIST[0]:
      bigD = dat
    else:
      bigD = bigD.append(dat,ignore_index=True)
  return bigD

def plotColumn(dat,colNum,numLIST,fig,ax,tName):
  #plot the column number colNum
  C=(.7,.7,.7)
  plt.subplot(3,3,colNum)
  ax[0,colNum-1] = sns.boxplot(x="N", y="Slope",fliersize=0,color=C, data=dat)
  plt.grid(True,axis='y')
  plt.ylim(0,2.1)
  plt.xlabel('')
  plt.title(tName)
  
  plt.subplot(3,3,3+colNum)
  ax[1,colNum-1] = sns.boxplot(x="N", y="C-index",fliersize=0,color=C, data=dat)
  plt.grid(True,axis='y')
  plt.ylim(0,1.1)
  plt.xlabel('')

  plt.subplot(3,3,6+colNum)
  ax[2,colNum-1] = sns.boxplot(x="N", y="CIL",fliersize=0,color=C,data=dat)
  plt.grid(True,axis='y')
  plt.ylim(0,1.5)

  ax[0,colNum-1].axes.xaxis.set_ticklabels([])
  ax[1,colNum-1].axes.xaxis.set_ticklabels([])
  ax[2,colNum-1].axes.xaxis.set_ticklabels(numLIST)
  if colNum>1:
    ax[0,colNum-1].axes.yaxis.set_ticklabels([])   
    ax[1,colNum-1].axes.yaxis.set_ticklabels([])   
    ax[2,colNum-1].axes.yaxis.set_ticklabels([])   
    ax[0,colNum-1].set_ylabel('')
    ax[1,colNum-1].set_ylabel('')
    ax[2,colNum-1].set_ylabel('')
    
  return


# MAIN
os.chdir(mydir)

bigD_bai = getZING('BAsmallZ','C-index',numLIST_bai)
bigD_cov = getZING('COsmallZ','C-index',numLIST_cov)
bigD_st = getZING('STsmallZ','C-index',numLIST_st)



print('plotting...')
fig, ax = plt.subplots(3,3,sharex='col',sharey='row',figsize=(8,8))
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.2)

plotColumn(bigD_bai,1,numLIST_bai,fig,ax,'BAI')
plotColumn(bigD_cov,2,numLIST_cov,fig,ax,'COVA')
plotColumn(bigD_st,3,numLIST_st,fig,ax,'ST')

plt.show()
print('saving...')
# jpeg in 300 dpi
fig.savefig('ZplotFull-v2.jpg',dpi=300)

