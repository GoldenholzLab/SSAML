#!/usr/bin/python
# SSAML
# This code is designed to calculate the number of patients (or events) needed for a clinical
# validation study of a machine learning algorithm. Choose your dataset wisely, because
# inadequate data will give you wrong answers. Use at your own risk.
# Daniel Goldenholz, MD, PhD

# USAGE:
#power.py <runMode> <iterNumber> <maxPts> <confint> <infile> <outdir> <peopleTF> <survivalTF> <resampReps> <bootReps>
# runMode - the mode we are running
#  = 1 iterate
#  = 2 summarize the local iterations
#  = 3 global summary
# iterNumber - the iteration number to run (between 0 and 9999)
# maxPts - how many patients are available in total
# confint - the confidence interval to compute (eg 0.995, .997 etc)
# infile - full path of the input data file
# outdir - full path of the output data directory
# peopleTF -- 1 means you want people. 0 means you want EVENTS.
# survivalTF - 0 if no survival stats, 1 if survival based stats
# resampReps = 40 if not listed, but how many outter reps to do
# bootReps = 1000 if not listed, how many inner reps to do

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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from lifelines import KaplanMeierFitter,CoxPHFitter
from tqdm import tqdm   # for showing progress bar


# INPUT PARAMETERS
if (len(sys.argv)<7):
  print('Need at least 6 args!')
  exit()
runMode = int(sys.argv[1])
iterNumber = int(sys.argv[2])
maxPts = int(sys.argv[3])
#confint options 0.95, 0.99, 0.999,...
confint = float(sys.argv[4])
big_file = sys.argv[5]
mydir= sys.argv[6]
# set some defaults if not all params available
if (len(sys.argv)>7):
  peopleTF = int(sys.argv[7])==1
  survivalTF = int(sys.argv[8]) == 1
  resampReps = int(sys.argv[9])
  bootReps = int(sys.argv[10])

else:
  # in this case, parameters were not given, so use assumed default values
  peopleTF = True
  survivalTF = False
  resampReps = 40
  bootReps = 1000

print('Running mode %d with survivalTF=%r peopleTF=%r iteration %d, maxpts %d, CI: %0.6f, resampReps=%d, bootReps=%d' % (runMode,survivalTF,peopleTF,iterNumber,maxPts,confint,resampReps,bootReps))
print('Input file = %s' % big_file)
print('Output directory = %s' % mydir)

# GLOBAL CONSTANT DEFINITIONS
# the double bootstrapping of SSAML is always done with replacement. If you feel that this should be
# without replacement, change this flag below.
withReplacement = True
# this flag is for doing ZING files that produces a figure. It makes more files, and therefore is optional.
doEXTRA=True
# if you want runMode 1 to run using parallel processing on a single computer, set this to True, and n_jobs as needed
do_parallel=True
n_jobs=4

# FUNCTION DEFINITIONS

def runOneSet_inner(N,uids,c,bootReps,withReplacement,peopleTF,survivalTF):
  # This is a subfunctoin of runOneSet, used to assist with parallel processing when available
  # inputs are derived from runOneSet

  # Build a subgroup of sampled uid numbers
  sub_uids = makeSubGroup(uids,N,withReplacement,peopleTF)
  resultX = []   # calcX sometimes throws error due to specific bootstrap indices, only keep good ones

  # bootstrap for bootRep number of times
  for boots in range(bootReps):
    # build a subgroup based on the subgroup from above (resampled with each iteration here)
    boot_subs = makeSubGroup(sub_uids,N,withReplacement,peopleTF)
    # use the indices of the indices of the subgroups to define this bootstrapped subgroup
    myBoot= uids[sub_uids[boot_subs]]

    # now build c_subset from portions of c that have ID = the IDs from myBoot
    # the following code allows for repeated samples of the same ID
    ids = []
    for i in range(len(myBoot)):
      ids.extend(np.where(c.ID==myBoot[i])[0])
    c_subset = c.iloc[ids].reset_index(drop=True)
    
    # assuming calcX can obtain a calculation, do so for this subgroup and append the result
    try:
      resultX.append(calcX(c_subset,c,survivalTF))
    except Exception as ee:
      continue

  return np.array(resultX)


def runOneSet(N,uids,c,resampReps,bootReps,fName,withReplacement,doEXTRA,confint,peopleTF,survivalTF,do_parallel,n_jobs):
  # given N patients and the data, bootstrap for different population sizes to estimate ranges
  # fName is the filename for output, such as 'boom' which becomes 'boom-01.csv'
  #  and 'boom-02.csv',...
  # peopleTF if True then use number of people, if False use number of events
  # survivalTF is a flag to use or not use survival stats type summary values
  # do_parallel - True if parallel processing should be run
  # n_jobs - how many jobs to use with Parallel command

  fn = fName
  fn2 = 'ZING' + fName 

  # Here we use a loop of resampReps for the outter loop.
  # The inner loop is run by the subfunction runOneSet_inner to facilitate parallel processing if available/desired.
  if do_parallel:
    with Parallel(n_jobs=n_jobs, verbose=False) as par:
      resultXs = par(delayed(runOneSet_inner)(N,uids,c,bootReps,withReplacement,peopleTF,survivalTF) for resamp in tqdm(range(resampReps)))
  else:
    resultXs = [runOneSet_inner(N,uids,c,bootReps,withReplacement,peopleTF,survivalTF) for resamp in tqdm(range(resampReps))]

  # now that the outer loop is done, store the result of each outer loop as a line in an output file f
  for resultX in resultXs:
    with open(fn, 'a') as f:
      for i in range(3):
        printConf(resultX,i,f,i==0,confint)
      print('',end='\n',file=f)
    # If doEXTRA is true, additional detailed data is stored in file f2
    if doEXTRA:
      with open(fn2, 'ab') as f2:
        np.savetxt(f2, resultX, delimiter=',', fmt='%0.3f')

  print('done.')
  return

def calcX_survival(c_subset,c):
  # SUBFUNCTION OF calcX for the case of survival analysis. 
  # Given c_subset and full data c
  # Returns slope, c-index and CIL

  # initialize the fitters
  cph = CoxPHFitter()
  cph2 = CoxPHFitter()

  # Using CPH, form an estimator based on the subset of the full dataset (c_subset)
  cph.fit(c_subset, duration_col='T', event_col='C', formula = "z", cluster_col='ID')

  ## 1. GET SLOPE
  # newz obtains the predicted z
  newz = cph.predict_log_partial_hazard(c)
  c['newz'] = newz
  # now refit, this time using the predicted z as the input, but fit to the full data c
  cph2.fit(c,duration_col='T',event_col='C', formula = 'newz', cluster_col='ID')
  # obtain the hazard ratios from the refit
  temp = np.log(cph2.hazard_ratios_)
  # the "slope" is the HR from the newz covariate in the model
  slope = temp.newz

  ## 2. GET C INDEX (based on cph fit from the subset)
  c_index = cph.concordance_index_

  ## 3. GET EXP / OBS ie calibration in the large, ie CIL
  kmf = KaplanMeierFitter() 
  kmf.fit(c_subset['T'], c_subset['C'],label='Kaplan Meier Estimate')
  h = np.array(kmf.predict(c_subset['T']))
  ph = np.array(cph.predict_partial_hazard(c_subset))
  surv=ph*h   # this is the survival based on KM estimate times hazard
  EXP = np.sum(1-surv)  # this is expected number of "dead"
  OBS= np.sum(cph.event_observed)  # this is observed number of "dead"
  CIL = EXP / OBS

  # return the 3 metrics
  return [slope,c_index,CIL]
    
def calcX_regular(c_subset):
  # SUBFUNCTION of calcX for a non-survival analysis case.
  # Given c_subset, compute slope, AUC and CIL

  # the column called 'event' will be predicted by probability 'p'
  y=c_subset['event']
  pred=c_subset['p']

  ## 1. GET SLOPE
  # given true y and predicted, calculate slope and bias of calibration curve
  fraction_of_positives, mean_predicted_value = calibration_curve(y, pred, n_bins=10)
  reg = LinearRegression().fit(np.reshape(mean_predicted_value,(-1,1)), fraction_of_positives)
  slope= reg.coef_[0]
  
  ## 2. GET AUC
  fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
  AUC = metrics.auc(fpr, tpr)

  ## 3. GET EXP/OBS  ie calibration in the large, CIL
  CIL = np.sum(pred) / np.sum(y)
  
  # here is the return value X
  return [slope,AUC,CIL]

def calcX(c_subset,c,survivalTF):
  # given the c_subset to work with and (when needed) the original c matrix, and a flag to use survival data,
  # we will calculate either the slope,AUC and CIL or slope,c-index and CIL
  if survivalTF==1:       # Cox proportional hazard data is requested here
    X = calcX_survival(c_subset,c)
  else:                   # This does not need survival data. It is based entirely on the subset.
    X = calcX_regular(c_subset)
  return X

def makeSubGroup(uids,howmany,withReplace,peopleTF):
  # given a set of unique ids, how many of them desired, choose some at random with or without replacement
  # output the indices, not the subgroup of samples
  if peopleTF==1:   # This is a people based analysis
    # request a set of uids of seize howmany
    sub_uids = np.random.choice(len(uids),size=howmany,replace=withReplace)
  else:             # This is an event based analysis
    # permute the uids
    perm = np.random.permutation(len(uids))
    # add up events within the permuted set
    cs = np.cumsum(c.event[perm])
    # find where we can cut to have enough events
    inds = np.where(cs>=howmany)
    if len(inds[0])==0:     # if there are not enough, just take them all
      thisManyPeople = len(uids)
    else:               # this is the normal case
      # now cut there
      thisManyPeople = inds[0][0]
      # keep that many people in the permuted list
    sub_uids = perm[0:thisManyPeople]

  return np.sort(sub_uids)

def printConf(arrX,ind,f,firstTF,cutoff):
  # arrX is NxP - choose ind for dim P. also f is the file handler.
  # given array of numbers, get conf interval, then print median, lower, upper
  # firstTF if true will omit leading ,

  arr = arrX[:,ind]   # the array
  marr = np.nanmean(arr) # the mean excluding any NaN

  # gaussian lower and upper confidence interval
  #cutoff suggestions = 0.68, 0.955, 0.997, 0.9999, 0.999999, etc
  CIlower,CIupper = norm.interval(cutoff, loc=marr, scale=np.std(arr))

  # if this isn't the first item, place a comma
  if firstTF==False:
    print(f',',end='',file=f)
  # write the mean and confidence interval requested with 3 significant digits
  print(f'{marr:0.3},{CIlower:0.3},{CIupper:0.3}',end='',file=f)
  # also return those 3 numbers as a numpy array
  theArr = np.array([marr,CIlower,CIupper])

  return theArr

def getSummary(fName,fullResultName,trueX,howmany,confint):
  # Read the CVS file which has all raw data.
  # Then compute RWD, BIAS and COVP for each of the 3 metrics (AUC, c-index, CIL)
  # Then output howmany patient/event, the confidence interval, ave. RWD, ave BIAS and COVP to fullFresultName
  # trueX is the ground truth for each metric used for comparison

  # now read in all the iterations
  P1 = pd.read_csv(fName,header=None)
  full_result = np.zeros((3,5))
  for subSCORE in range(3):
    theTrue = trueX[subSCORE]
    theThree = np.array(P1.iloc[:,(3*subSCORE):(3*subSCORE+3)])
    # RWD: get ave RWD. RWD = CI / true
    rwds = np.array((theThree[:,2] - theThree[:,1])/theTrue)
    ave_rwd = np.mean(rwds[np.isfinite(rwds)])
    #ave_rwd = np.average(rwds)
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

    full_result[:,subSCORE+2] = [ave_rwd,ave_bias,COVP]

  # only 3 digits interesting
  full_result = np.floor(1000*full_result) / 1000
  # put in the howmany and confint, which don't need only 3 digits
  full_result[:,0] = howmany
  full_result[:,1] = confint
  # write the output of the full result
  pd.DataFrame(full_result).to_csv(fullResultName,header=None, index=None,float_format='%0.6f')
  return  

def showSummary(rwd,bias,covp,numLIST,oldALL,survivalTF): 
  # This function is used to print out the summarized results

  # if using survival data, calcX used C-index. Otherwise used AUC. Technically c-index encompasses both.
  if survivalTF==True:
    useme = 'C-index'
  else:
    useme = 'AUC'

  # read in the files with the 3 summary stats from RWD, BIAS and COVP
  R = pd.read_csv(rwd,delimiter=',',header=None)
  B = pd.read_csv(bias,delimiter=',',header=None)
  C = pd.read_csv(covp,delimiter=',',header=None)

  # Compose a pandas dataframe
  R.columns = ['howmany','confint','RWD slope','RWD ' + useme,'RWD CIL']
  numLIST = R['howmany']
  R = R.drop('howmany',axis=1)
  B.columns = ['howmany','confint','BIAS slope','BIAS ' + useme,'BIAS CIL']
  B = B.drop(['howmany','confint'],axis=1)
  C.columns = ['howmany','confint','COVP slope','COVP ' + useme,'COVP CIL']
  C = C.drop(['howmany','confint'],axis=1)
  ALL = pd.concat([R,B,C],axis=1)
  ALL.index = numLIST

  # Print out the dataframe
  print('RWD goal < 0.5, BIAS goal < 5%, COVP > 95%')
  print(ALL.transpose())

  # Set the dataframe to nan if appropriate entries do not meet COVP>=95% criteria  
  ALL[ALL['COVP slope']<0.95] = np.nan
  ALL[ALL['COVP ' + useme]<0.95]=np.nan
  ALL[ALL['COVP CIL']<0.95]=np.nan

  ALL = ALL.transpose()
  # here, keep all the entries that don't have nan
  oldALL[np.isnan(ALL)==0] = ALL[np.isnan(ALL)==0]

  return oldALL

def plotZING(prefixN,numLIST,survivalTF):
  # the ZING files allow the detailed boxplots or violin plots to be displayed

  # if using survival data, calcX used C-index. Otherwise used AUC. Technically c-index encompasses both.
  if survivalTF==True:
    useme = 'C-index'
  else:
    useme = 'AUC'

  # fill a pandas dataframe with slope, c-index/auc and cil
  for howmany in numLIST:
    fn = prefixN + str(howmany).zfill(4) + '.csv'
    dat = pd.read_csv(fn,sep=',',header=None)
    dat.columns =['Slope',useme,'CIL']
    dat['N'] = dat.Slope*0 + howmany
    if howmany == numLIST[0]:
      bigD = dat
    else:
      bigD = bigD.append(dat,ignore_index=True)
  
  # draw a boxplot for each of the 3 metrics, save as a file with 300 dpi
  fig, (ax1, ax2, ax3) = plt.subplots(3,1)
  plt.subplot(3,1,1)
  ax1 = sns.boxplot(x="N", y="Slope", data=bigD,showfliers=False)
  plt.subplot(3,1,2)
  ax2 = sns.boxplot(x="N", y=useme, data=bigD,showfliers=False)
  plt.subplot(3,1,3)
  ax3 = sns.boxplot(x="N", y="CIL",data=bigD,showfliers=False)
  fig.savefig('Zplot.jpg',dpi=300)
  return


# MAIN

T1= time.time()
os.chdir(mydir)

# read in data file and check format requirements.
c = pd.read_csv(big_file, sep=',')
assert c.shape[0] > 0, "No rows in input csv file detected. Check if file is empty or wrongly formatted."

if survivalTF:
  cols_required = ['ID', 'T', 'C', 'z']
  assert all(np.isin(cols_required, c.columns)), f"Survival analysis type of data specified (survivalTF==True), \
  input file does not contain required columns (required: {cols_required}, contained: {c.columns}."

  assert peopleTF==1, "For survival analysis type of data (survivalTF==True), peopleTF==1 is expected."

else:
  cols_required = ['ID', 'event', 'p']
  assert all(np.isin(cols_required, c.columns)), f"Non-survival analysis type of data specified (survivalTF==False), \
  input file does not contain required columns (required: {cols_required}, contained: {c.columns}."

uids = pd.unique(c.ID)
howmany = maxPts

if runMode==1:
  # RUN MODE 1 - submit a set of iterations, possibly for a supercomputer cluster
  fName = 'num' + str(howmany).zfill(4) + str(iterNumber).zfill(4) + '_' + str(confint) + '.csv'
  runOneSet(howmany,uids,c,resampReps,bootReps,fName,withReplacement,doEXTRA,confint,peopleTF,survivalTF,do_parallel,n_jobs)
  
if runMode==2:
  # RUN MODE 2
  # to clean up after runmode 1. Here we calculate the RWD, BIAS and COVP from raw data.
  fName = 'num' + str(howmany).zfill(4) +  '_' + str(confint) + '.csv'
  fullResultName = 'full' + str(howmany).zfill(4) +  '_' + str(confint) + '.csv'
  trueX = calcX(c,c,survivalTF)
  getSummary(fName,fullResultName,trueX,howmany,confint)
  
if runMode==3:
  # RUN MODE 3
  # Use summary data to print and plot

  x = pd.read_csv('conflist.setup',delimiter=' ',header=None)
  assert x.shape[0] > 0, "No rows in input conflist.setu file detected. Check if file is empty or wrongly formatted."
  clist=np.array(x.iloc[0,])

  x = pd.read_csv('RWD_' + str(clist[0]) +'.txt',delimiter=',',header=None)
  x.columns = ['howmany','confint','RWD slope','RWD C-index','RWD CIL']
  numLIST = np.array(x.howmany)
  numLIST = numLIST.astype(int)

  temp=np.empty((10,len(numLIST)))
  temp[:]=np.nan
  if survivalTF==True:
    useme = 'C-index'
  else:
    useme = 'AUC'

  ALL = pd.DataFrame(temp,index=['confint','RWD slope','RWD ' + useme,'RWD CIL','BIAS slope','BIAS ' + useme,'BIAS CIL','COVP slope','COVP ' + useme,'COVP CIL'],columns=numLIST)
  for confint in reversed(clist):
    ALL = showSummary('RWD' + '_' + str(confint) + '.txt','BIAS' + '_' + str(confint) + '.txt','COVP' + '_' + str(confint) + '.txt',numLIST,ALL,survivalTF)
  print('The frankenstein is...')
  print(ALL)
  print('The columnn with numbers (i.e. no NaN values) represents the lowest sample size selected by SSAML.')
  try:
    plotZING('smallZ',numLIST,survivalTF)
  except:
    print('Plot ZING files not performed.')
  

# in case record keeping is important for supercomputer time, report the run duration here
T2 = time.time()
print('Runtime = %0.1f' % (T2-T1))
