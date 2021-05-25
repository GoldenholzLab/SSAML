#!/usr/bin/python

# First load up the libraries needed
# this for drawing headless
import matplotlib
from matplotlib.ticker import StrMethodFormatter

#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import os
import sys
import time


# constant definitions

def getZING(prefixN,middleOne,numLIST):
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
# prepare big file
T1= time.time()
mydir = '/Users/danisized/iCloud/Latest/power-calculator-stat/Zfiles'

os.chdir(mydir)
numLIST = [500,1000,1500,2000]
bigD_bai = getZING('BAsmallZ','C-index',numLIST)
numLIST = [125, 150, 175, 200]
bigD_cov = getZING('COsmallZ','C-index',numLIST)
numLIST=[20,40,60,80]
bigD_st = getZING('STsmallZ','C-index',numLIST)



print('plotting...')
fig, ax = plt.subplots(3,3,sharex='col',sharey='row',figsize=(8,8))
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.2)

numLIST = [500,1000,1500,2000]
plotColumn(bigD_bai,1,numLIST,fig,ax,'BAI')
numLIST = [125, 150, 175, 200]
plotColumn(bigD_cov,2,numLIST,fig,ax,'COVA')
numLIST=[20,30,40,50]
plotColumn(bigD_st,3,numLIST,fig,ax,'ST')

plt.show()
print('saving...')
fig.savefig('ZplotFull.jpg',dpi=300)



T2 = time.time()
print('Runtime = %0.1f' % (T2-T1))