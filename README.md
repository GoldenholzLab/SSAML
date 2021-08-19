# SSAML
 SSAML: sample size analysis for machine learning clinical validation studies
Question or comments: daniel.goldenholz@bidmc.harvard.edu
Daniel Goldenholz, 2021

HOW TO USE THIS TOOL:

1. edit runner_power depending on your dataset
2. edit run_power depending on the needs of your supercomputer cluster resources
3. If you have parallel processing capabilities, you can set the flag called do_parallel in power.py to True
4. If assumed features of the data or analysis require changes, make these in power.py

WHAT ARE THE IMPORTANT FILES:
1. runner_power
This is a runner script that will submit a sequence of commands. If using a supercomputer, this
script will submit multiple jobs.
You will need to edit this script to submit the jobs that are relevant for your task.

2. run_power
This is the wrapper script for power.py. It is necessary for submitting a supercomputer job. It contains
flags for the cluster to properly allocate resources when running power.py.
If you are not running a supercomputer cluster, run_power will still work fine, and it simply becomes a
convienence wrapper.

3. power.py
This is the main python code to run SSAML. The wrappers above make it easier to use this script.


Info about the 3 datasets we used:
# 0: ST dataset, with repeated samples from same patients, ID is already a field in the database, goal is PATIENTS
#     there is an assumed column called "szTF" which will be 1 for true and 0 for false, this will be ground truth
#     there is an assumed column called "AI" which will be a fraction 0..1 for the predicted value
#     there is an assumed column called "ID" which has a unique ID number for each patient (multiple entries per patient ok)
#     the columns assumed:'ID','szTF','AI','RMR'. One row per entry. This is a CSV file
# 1: COVA dataaset, single sample per patient, goal is number of EVENTS not PATIENTS
#     we assume columns are present: ['actual','Prob-dead','Prob-ICU-MV','Prob-Hosp']
#     the 'actual' column is the ground truth. The sum of the other Prob columns divided by 100 is assumed to be
#     0..1 probability prediction
# 2: BAI dataset, longitudinal survival data, goal is number of PATIENTS
#    this assumes a 'z','T',and 'C' columns. z is the z-score value, a covariate for Cox proportional hazard.
#    T is time in years, and C is censored (1=yes, 0=no).