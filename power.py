#!/usr/bin/python
# This code is designed to calculate the number of patients (or events) needed for a clinical
# validation study of a machine learning algorithm. Choose your dataset wisely, because
# inadequate data will give you wrong answers. Use at your own risk.
# Daniel Goldenholz, MD, PhD

# USAGE:
#power.py <runMode> <dataTYPE> <iterNumber> <maxPts> <confint> <infile> <outdir> <peopleTF> <survivalTF>
# runMode - the mode we are running
#  = 1 iterate
#  = 2 summarize the local iterations
#  = 3 global summary
# dataTYPE = 0 for ST type database. 1 for COVA, 2 for BAI, 3 for anything else.
#      note: 0,1 and 2 will override the peopleTF and survivalTF options.
# iterNumber - the iteration number to run (between 0 and 9999)
# maxPts - how many patients are available in total
# confint - the confidence interval to compute (eg 0.995, .997 etc)
# infile - full path of the input data file
# outdir - full path of the output data directory
# peopleTF -- 1 means you want people. 0 means you want EVENTS.
# survivalTF - 0 if no survival stats, 1 if survival based stats

# First load up the libraries needed
# this for drawing headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import sys
from joblib import Parallel, delayed
import time
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from lifelines import KaplanMeierFitter,CoxPHFitter
from tqdm import tqdm   # for showing progress bar


# INPUTS
if (len(sys.argv)<8):
  print('Need at least 7 args!')
  exit()
runMode = int(sys.argv[1])
dataTYPE = int(sys.argv[2])
iterNumber = int(sys.argv[3])
maxPts = int(sys.argv[4])
#confint options 0.95, 0.99, 0.999,...
confint = float(sys.argv[5])
big_file = sys.argv[6]
mydir= sys.argv[7]
if (len(sys.argv)>8):
  peopleTF = int(sys.argv[8])==1
  survivalTF = int(sys.argv[9]) == 1
else:
  peopleTF = True
  survivalTF = False
# dataTYPE:
# 0: ST dataset, with repeated samples from same patients, ID is already a field in the database, goal is PATIENTS
# 1: COVA dataaset, single sample per patient, goal is number of EVENTS not PATIENTS
# 2: BAI dataset, longitudinal survival data, goal is number of PATIENTS

print('Running mode %d with survivalTF=%r peopleTF=%r iteration %d, maxpts %d, CI: %0.6f' % (runMode,survivalTF,peopleTF,iterNumber,maxPts,confint))
print('Input file = %s' % big_file)
print('Output directory = %s' % mydir)

# CONSTANT DEFINITIONS
withReplacement = True
resampReps=40
bootReps=1000
# this flag is for doing ZING files that produces a figure. It makes more files, and therefore is optional.
doEXTRA=True


# FUNCTION DEFINITIONS


def runOneSet_inner(N,uids,c,resampReps,bootReps,fName,withReplacement,doEXTRA,confint,peopleTF,survivalTF):
  sub_uids = makeSubGroup(uids,N,withReplacement,peopleTF)
  resultX = []
  for boots in range(bootReps):
    boot_subs = makeSubGroup(sub_uids,N,withReplacement,peopleTF)
    myBoot= uids[sub_uids[boot_subs]]
    # the following code allows for repeated samples of the same ID
    ids = []
    for i in range(len(myBoot)):
      ids.extend(np.where(c.ID==myBoot[i])[0])
    c_subset = c.iloc[ids].reset_index(drop=True)
    resultX.append( calcX(c_subset,c,survivalTF) )
  return np.array(resultX)


def runOneSet(N,uids,c,resampReps,bootReps,fName,withReplacement,doEXTRA,confint,peopleTF,survivalTF):
  # given N patients and the data, bootstrap for different population sizes to estimate ranges
  # fName is the filename for output, such as 'boom' which becomes 'boom-01.csv'
  #  and 'boom-02.csv',...
  # peopleTF if True then use number of people, if False use number of events
  # survivalTF is a flag to use or not use survival stats type summary values

  fn = fName
  fn2 = 'ZING' + fName
  
  parallel = False
  n_jobs = 1  # number of processors
  if parallel:
    with Parallel(n_jobs=n_jobs, verbose=False) as par:
      resultXs = par(delayed(runOneSet_inner)(N,uids,c,resampReps,bootReps,fName,withReplacement,doEXTRA,confint,peopleTF,survivalTF) for resamp in tqdm(range(resampReps)))
      
  else:
    resultXs = [runOneSet_inner(N,uids,c,resampReps,bootReps,fName,withReplacement,doEXTRA,confint,peopleTF,survivalTF) for resamp in tqdm(range(resampReps))]
    
  for resultX in resultXs:
    with open(fn, 'a') as f:
      for i in range(3):
        printConf(resultX,i,f,i==0,confint)
      print('',end='\n',file=f)
    # save extra data
    if doEXTRA:
      with open(fn2, 'ab') as f2:
        np.savetxt(f2, resultX, delimiter=',', fmt='%0.3f')

  print('done.')
  return


def calcX(c_subset,c,survivalTF):
  # given the c_subset to work with and (when needed) the original c matrix, and a flag to use survival data,
  # we will calculate either the slope,AUC and CIL or slope,c-index and CIL

  if survivalTF==1:
    cph = CoxPHFitter()
    cph2 = CoxPHFitter()

    cph.fit(c_subset, duration_col='T', event_col='C', formula = "z")

    # GET SLOPE
    newz = cph.predict_log_partial_hazard(c)
    c['newz'] = newz
    cph2.fit(c,duration_col='T',event_col='C', formula = 'newz')
    temp = np.log(cph2.hazard_ratios_)
    slope = temp.newz

    # GET C INDEX
    c_index = cph.concordance_index_

    # GET EXP / OBS ie calibration in the large, CIL
    kmf = KaplanMeierFitter() 
    kmf.fit(c_subset['T'], c_subset['C'],label='Kaplan Meier Estimate')
    h = np.array(kmf.predict(c_subset['T']))
    ph = np.array(cph.predict_partial_hazard(c_subset))
    surv=ph*h
    EXP = np.sum(1-surv)
    OBS= np.sum(cph.event_observed)
    CIL = EXP / OBS

    X = [slope,c_index,CIL]
  else:
    y=c_subset['event']
    pred=c_subset['p']

    # GET SLOPE
    # given true y and predicted, calculate slope and bias of calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y, pred, n_bins=10)
    reg = LinearRegression().fit(np.reshape(mean_predicted_value,(-1,1)), fraction_of_positives)
    slope= reg.coef_[0]
    
    # GET AUC
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    AUC = metrics.auc(fpr, tpr)

    # GET EXP/OBS  ie calibration in the large, CIL
    CIL = np.sum(pred) / np.sum(y)

    X = [slope,AUC,CIL]

  return X

def makeSubGroup(uids,howmany,withReplace,peopleTF):
  # given a set of unique ids, how many of them desired, choose some at random with or without replacement
  # output the indices, not the subgroup of samples
  if peopleTF==1:
    sub_uids = np.random.choice(len(uids),size=howmany,replace=withReplace)
  else:
    perm = np.random.permutation(len(uids))
    cs = np.cumsum(c.event[perm])
    inds = np.where(cs>=howmany)
    thisManyPeople = inds[0][0]
    sub_uids = perm[0:thisManyPeople]

  return np.sort(sub_uids)

def printConf(arrX,ind,f,firstTF,cutoff):
  # arrX is NxP - choose ind for dim P. also f is the file handler.
  # given array of numbers, get conf interval, then print median, lower, upper
  # firstTF if true will omit leading ,

  arr = arrX[:,ind]
  #marr = np.nanmean(arr)

  # gaussian 95% CI
  #cutoff options = 0.68, 0.955, 0.997, 0.9999, 0.999999, etc
  #CIlower,CIupper = norm.interval(cutoff, loc=marr, scale=np.std(arr))
  # if you wanted to use NONPARAMETRIC CI...
  # CIlower = np.percentile(arr,2.5)
  # CIupper = np.percentile(arr,97.5)
  marr, CIlower, CIupper = np.percentile(arr, (50, (1-cutoff)/2*100, (1+cutoff)/2*100))

  if firstTF==False:
    print(f',',end='',file=f)
  print(f'{marr:0.3},{CIlower:0.3},{CIupper:0.3}',end='',file=f)
  theArr = np.array([marr,CIlower,CIupper])

  return theArr

def getSummary(fName,fullResultName,trueX,howmany,confint):
  # now read in all the iterations
  P1 = pd.read_csv(fName,header=None)
  full_result = np.zeros((3,5))
  for subSCORE in range(3):
    theTrue = trueX[subSCORE]
    theThree = np.array(P1.iloc[:,(3*subSCORE):(3*subSCORE+3)])
    # RDW: get ave RDW. RDW = CI / true
    rdws = np.array((theThree[:,2] - theThree[:,1])/theTrue)
    ave_rdw = np.mean(rdws[np.isfinite(rdws)])
    #ave_rdw = np.average(rdws)
    # BIAS: get ave BIAS. BIAS = (true - est)/true
    biases = np.array((theTrue - theThree[:,0])/theTrue)
    #ave_bias = np.average(biases)
    ave_bias = np.mean(biases[np.isfinite(biases)])
    # COVP: percentage of time truth is in CONF INT
    TF1 = theTrue>=theThree[:,1]
    TF2 = theTrue<=theThree[:,2]
    TFs = np.array(0.0+(np.logical_and(TF1,TF2)))
    COVP = np.mean(TFs[np.isfinite(TFs)])
    #COVP = np.average(TFs)    

    full_result[:,subSCORE+2] = [ave_rdw,ave_bias,COVP]

  # only 3 digits interesting
  full_result = np.floor(1000*full_result) / 1000
  # put in the howmany and confint, which don't need only 3 digits
  full_result[:,0] = howmany
  full_result[:,1] = confint
  # write the output of the full result
  pd.DataFrame(full_result).to_csv(fullResultName,header=None, index=None,float_format='%0.6f')
  return  

def showSummary(rwd,bias,covp,numLIST,oldALL,survivalTF): 
  if survivalTF==True:
    useme = 'C-index'
  else:
    useme = 'AUC'

  R = pd.read_csv(rwd,delimiter=',',header=None)
  B = pd.read_csv(bias,delimiter=',',header=None)
  C = pd.read_csv(covp,delimiter=',',header=None)

  R.columns = ['howmany','confint','RDW slope','RWD ' + useme,'RWD CIL']
  numLIST = R['howmany']
  R = R.drop('howmany',axis=1)
  B.columns = ['howmany','confint','BIAS slope','BIAS ' + useme,'BIAS CIL']
  B = B.drop(['howmany','confint'],axis=1)
  C.columns = ['howmany','confint','COVP slope','COVP ' + useme,'COVP CIL']
  C = C.drop(['howmany','confint'],axis=1)
  ALL = pd.concat([R,B,C],axis=1)
  ALL.index = numLIST

  print('RWD goal < 0.5, BIAS goal < 5%, COVP < 95%')
  print(ALL.transpose())
  
  ALL[ALL['COVP slope']<0.95] = np.nan
  ALL[ALL['COVP ' + useme]<0.95]=np.nan
  ALL[ALL['COVP CIL']<0.95]=np.nan
  ALL = ALL.transpose()
  oldALL[np.isnan(ALL)==0] = ALL[np.isnan(ALL)==0]

  return oldALL

def plotZING(prefixN,numLIST,survivalTF):
  if survivalTF==True:
    useme = 'C-index'
  else:
    useme = 'AUC'

  for howmany in numLIST:
    fn = prefixN + str(howmany).zfill(4) + '.csv'
    dat = pd.read_csv(fn,sep=',',header=None)
    dat.columns =['Slope',useme,'CIL']
    dat['N'] = dat.Slope*0 + howmany
    if howmany == numLIST[0]:
      bigD = dat
    else:
      bigD = bigD.append(dat,ignore_index=True)
  

  fig, (ax1, ax2, ax3) = plt.subplots(3,1)
  plt.subplot(3,1,1)
  ax1 = sns.boxplot(x="N", y="Slope", data=bigD)
  plt.subplot(3,1,2)
  ax2 = sns.boxplot(x="N", y=useme, data=bigD)
  plt.subplot(3,1,3)
  ax3 = sns.boxplot(x="N", y="CIL",data=bigD)
  fig.savefig('Zplot.jpg',dpi=300)
  return


# MAIN
# prepare big file
T1= time.time()
os.chdir(mydir)
if dataTYPE==0:
  # ST datafile
  c = pd.read_csv(big_file,sep=',',names=['ID','szTF','AI','RMR'])
  uids = pd.unique(c.ID)
  c.rename(columns={'szTF':'event'},inplace=True)
  c.rename(columns={'AI':'p'},inplace=True)
  peopleTF=True
  survivalTF=False
elif dataTYPE==1:
  # COVA datafile
  c = pd.read_csv(big_file,sep=',')
  uids = np.array(range(c.shape[0]))
  c['ID'] = uids
  AInames= ['Prob-dead','Prob-ICU-MV','Prob-Hosp']
  c['p'] = (c[AInames[0]] + c[AInames[1]] + c[AInames[2]])/100
  c['event'] = 0.0 + (c['actual']>0)
  peopleTF=False
  survivalTF=False
elif dataTYPE==2:
  c = pd.read_csv(big_file,sep=',')
  uids =  uids = np.array(range(c.shape[0]))
  c['ID'] = uids
  peopleTF=True
  survivalTF=True
else:
  c = pd.read_csv(big_file,sep=',')
  uids = np.array(range(c.shape[0]))
  c['ID'] = uids

howmany = maxPts

if runMode==1:
  fName = 'num' + str(howmany).zfill(4) + str(iterNumber).zfill(4) + '_' + str(confint) + '.csv'
  runOneSet(howmany,uids,c,resampReps,bootReps,fName,withReplacement,doEXTRA,confint,peopleTF,survivalTF)
  
if runMode==2:
  # to clean up after runmode 1
  fName = 'num' + str(howmany).zfill(4) + str(iterNumber).zfill(4) + '_' + str(confint) + '.csv'
  fullResultName = 'full' + str(howmany).zfill(4) +  '_' + str(confint) + '.csv'
  trueX = calcX(c,c,survivalTF)
  getSummary(fName,fullResultName,trueX,howmany,confint)
  
if runMode==3:
  x = pd.read_csv('conflist.setup',delimiter=' ',header=None)
  clist=np.array(x.iloc[0,])
  x = pd.read_csv("RWD_0.955.txt",delimiter=',',header=None)
  x.columns = ['howmany','confint','RDW slope','RWD C-index','RWD CIL']
  numLIST = np.array(x.howmany)
  numLIST = numLIST.astype(int)

  temp=np.empty((10,len(numLIST)))
  temp[:]=np.nan
  if survivalTF==True:
    useme = 'C-index'
  else:
    useme = 'AUC'

  ALL = pd.DataFrame(temp,index=['confint','RDW slope','RWD ' + useme,'RWD CIL','BIAS slope','BIAS ' + useme,'BIAS CIL','COVP slope','COVP ' + useme,'COVP CIL'],columns=numLIST)
  for confint in reversed(clist):
    ALL = showSummary('RWD' + '_' + str(confint) + '.txt','BIAS' + '_' + str(confint) + '.txt','COVP' + '_' + str(confint) + '.txt',numLIST,ALL,survivalTF)
  print('The frankenstein is...')
  print(ALL)
  plotZING('smallZ',numLIST,survivalTF)


#!head -n 1 $fullResultName >> RWD.txt
#!head -n 2 $fullResultName | tail -n 1 >> BIAS.txt
#!tail -n 1 $fullResultName >> COVP.txt 

T2 = time.time()
print('Runtime = %0.1f' % (T2-T1))
