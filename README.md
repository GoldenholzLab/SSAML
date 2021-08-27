# SSAML
SSAML: sample size analysis for machine learning clinical validation studies

Question or comments: daniel.goldenholz@bidmc.harvard.edu

Daniel Goldenholz, 2021


HOW TO USE THIS TOOL:
-------------------
1. Preprocess: Prepare your dataset, SSAML expects a .csv file with certain column names (see below).
2. Edit runner_power: data filepath, specify SSAML parameters.
3. Optional: If you have supercomputer/parallel processing capabilities: edit run_power.sh.
4. Optional: If you have parallel processing capabilities, you can set the flag called do_parallel in power.py to True.
5. Execute runner_power.sh with runMode=1, followed by runMode=2 to obtain SSAML results.

In runner_power, user has to set the filepath to the dataset to be analyzed, and the output directory. Further, SSAML parameters can be specified:  
-- peopleTF: 1 if patient based, 0 if event based. (default 1)  
-- survivalTF: 1 for survival analysis/dataset, 0 if not (default 0).  
-- resampReps: number of repitions (outer loop) to be performed (default 1000).  
-- bootReps: bootstrap repitions in inner loop (default 40)  

In runner_power, user can also save/shortcut specific filepath and parameter settings and run such settings when setting the paramsCONFIG parameter, see code. This is a convenience parameter only, not required to modify. The code currently contains the following parameter setting shortcuts:  
--paramsCONFIG=0: Default mode. custom dataset, must be a csv file (see below).  
--paramsCONFIG=1: Parameters used for the seizure prediction task (ST dataset), as presented in the paper.  
--paramsCONFIG=2: Parameters used for the COVID risk prediction task (COVA dataset), as presented in the paper.  
--paramsCONFIG=3: Parameters used for the longitudinal survival analysis task (BAI mortality dataset), as presented in the paper.  


DATA PREPARATION
-------------------
The format of the .csv file expected by the AASM code differs slightly for non-survival-analysis and survival-analysis tasks:

##### 'regular', non-survival analysis model.  
columns:  
-- ID: unique patient identifier (integers)  
-- event: ground truth / label (integers)  
-- p: model output, event probability  
rows are data observations (i.e. one row per event/patient)  

##### survival analysis model.  
columns:  
-- ID: unique patient identifier (integers)  
-- C: censhorhip information (i.e. 1 for censored, 0 for not censored)  
-- z is the z-score value, a covariate for Cox proportional hazard.  
-- T: time to event  
rows are data observations (i.e. one row per event/patient)

Useful to review: data_format_preprocess.ipynb  
This is a Jupyter Notebook that allows for preprocessing and formatting of the main assumed files types.
Very simple fake files (sample_data_bai_mortality.csv, sample_data_cova.csv, sample_data_st.csv) are produced here with no statistical meaning just to demonstrate how the files are
organized. It can be helpful to look at this notebook to better understand the assumed structure of the 
input files.


WHAT ARE THE IMPORTANT FILES:
-------------------
1. runner_power.sh

This is a runner script that will submit a sequence of commands. If using a supercomputer, this
script will submit multiple jobs.
You will need to edit this script to submit the jobs that are relevant for your task.
For example, if you want to use a supercomputer, change use_supercomputer to 1. It is set to 0 by default (ie. no supercomputer).
NOTE: some files are produced in bash scripting in runner_power that power.py requires.
If you choose NOT to use runner_power, make sure to produce analogous files.

2. run_power.sh

This is the wrapper script for power.py. It is necessary for submitting a supercomputer job. It contains
flags for the cluster to properly allocate resources when running power.py.
If you are not running a supercomputer cluster, run_power will still work fine, and it simply becomes a
convienence wrapper.

3. power.py

This is the main python code to run SSAML. The wrappers above make it easier to use this script.


Additional files that are not required for SSAML, but were used for our study
--------------------
1. get_interactive

this was a shortcut script used for the supercomputer cluster (not required)

2. make-power-pix

This brief script will produce the figure 2 plot from our paper, showing the distribution of metrics
from the many bootstrapped samples. It is not required for SSAML use.

3. make_fake_data.py

This was used to produce data files that can be used with SSAML as an example.
The fake file has a statistical relationship between the predicted and actual so that the details of the SSAML algorithm
can be further explored. 

4. COVA-FAKE.csv

this is a fake COVA file which has the proper formatting so that one can test out SSAML with a COVA style dataset.
This fake file has a statistical relationship between the predicted and actual so that the details of the SSAML algorithm
can be further explored. 
We did not provide the true COVA file because it was derived from protected health information.

Mini-Tutorial
--------------------
You can test the SSAML code with the COVA-FAKE.csv data. Simply modify infile path (line 75) to the COVA-FAKE.csv data, set an arbitrary outdir path (line 76) and run the code. To speed things up, you can also reduce the number of outer loops (parameter bootReps, line 80) temporarily.

