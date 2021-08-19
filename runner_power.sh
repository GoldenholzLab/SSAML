#!/bin/bash
# This is the master runner script for SSAML.
# From here, you will define where your data is. You will also specify what data type you have
# and various parameters of SSAML to run.
# A master runner script like this can be most helpful when using a supercomputer cluster,
# because it can send a series of individual job requests in to the cluster.
#
# USAGE:
#  runner_power.sh <runMode> <dataTYPE>
#   runMode:
#       1: used to submit a large batch of calculations
#       2: used to summarize the large batch when they are done. Mode 3 will automatically run afterwards.
#       3: used to produce the final output after mode 2 has summarized data.
#   dataTYPE:
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
# TO MODIFY THIS RUNNER SCRIPT FOR YOUR OWN USE:
#  First, determine what dataset type you have. If similar to one of the example datasets (ST, COVA, BAI), then 
#  you can use one of the dataTYPE shortcuts above. If not, define a different dataTYPE that sets the appropriate
#  variables below in the section calle **DATATYPE**

# Input parameters
runMode=$1
dataTYPE=$2

# constants
p=`pwd`

## *** DATATYPE ***  shortcut to set up variables for SSAML based on data type
# Notes:
#  infile - where your input file will be found
#  outdir - where output files should be saved
#  peopleTF - 1 if patient based, 0 if event based
#  survivalTF - 1 if survival analysis required, 0 if not
#  resampReps - number of repitions to be performed by power.py
#  ilist - a list of what iteration numbers to run, where each iteration has resampReps iterations within (between 0 and 9999)
#  maxlist - a list of candidate number of patients/events to test for sample szie power (any size ok)
#  conflist - a list of candidate confidence range (e.g. .95 = 95% confidence range) to test (any size ok)
case $dataTYPE in
    0)
        infile='/home/dmg16/deepMan/OUTPUT_for_paper/holdoutisThisReal_v2.csv'
        outdir='/home/dmg16/SSAML/OUTst'
        peopleTF=1
        survivalTF=0
        resampReps=40
        ilist=`seq 0 1 24`
        maxlist='20 40 60 80'
        conflist='0.955 0.997 0.9999 0.999999'
        ;;
    1)
        infile='/home/dmg16/SSAML/risk_7day-simplified.csv'
        outdir='/home/dmg16/SSAML/OUTcova'
        peopleTF=0
        survivalTF=0
        resampReps=40
        ilist=`seq 0 1 24`
        maxlist='125 150 175 200'
        conflist='0.955 0.997 0.9999 0.999999'
        ;;
    2)
        infile='/home/dmg16/SSAML/BA_For_Daniel.csv'
        outdir='/home/dmg16/SSAML/OUTbai'
        peopleTF=1
        survivalTF=1
        resampReps=10
        ilist=`seq 0 1 99`
        maxlist='500 1000 1500 2000'
        conflist='0.955 0.997 0.9999 0.999999'
        ;;
    *)
        echo "ERROR. No datatype specified."
        exit 1
        ;;
esac

# reset this for runMode 2 and 3
iterNumber=0

case $runMode in
    1)
        # This is runMode 1 - submit batches of scripts here (several hours)
        runMode=1
        mkdir -p $outdir/holder
        mkdir -p $outdir/theZING
        for confint in $conflist; do
            for maxPts in $maxlist; do
                for iterNumber in $ilist; do       
                    sbatch run_power.sh $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps
                done
            done
        done
        cd $outdir
        echo $conflist > conflist.setup
        ;;
    2)
        # Runmode 2, summarize the batches of data
        runMode=2
        cd $outdir
        for confint in $conflist; do
            for maxPts in $maxlist; do
                maxP=`printf "%04d" $maxPts`
                FILE=num${maxP}_${confint}.csv
                if [ ! -f "$FILE" ]; then
                    cat num${maxP}????_${confint}.csv > $FILE
                    mv num${maxP}????_${confint}.csv holder/
                fi
                echo "python $p/power.py $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps"
                python $p/power.py $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps
                fullResultName="full${maxP}_${confint}.csv"
                head -n 1 $fullResultName >> RWD_${confint}.txt
                head -n 2 $fullResultName | tail -n 1 >> BIAS_${confint}.txt
                tail -n 1 $fullResultName >> COVP_${confint}.txt
                FILE=smallZ${maxP}.csv
                if [ ! -f "$FILE" ]; then
                    cat ZINGnum${maxP}????_*.csv > $FILE
                    mv ZINGnum${maxP}????_*.csv theZING/
                fi
            done    
        done
        # do runMode3 automatically when runMode2 is done
        runMode=3
        cd $outdir
        maxPts=0
        confint=0
        python $p/power.py $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps
        ;;
    3)
        runMode=3
        cd $outdir
        maxPts=0
        confint=0
        python $p/power.py $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps
        ;;
    *)
        echo "ERROR. Only runMode 1,2,3 are allowed."
        ;;
esac




