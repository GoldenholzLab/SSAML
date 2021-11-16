#!/bin/bash
# if only type in runner_power, or runner_power-h, the usage is printed

Help()
{
echo "This is the master runner script for SSAML.
From here, you will define where your data is. You will also specify what data type you have
and various parameters of SSAML to run.
A master runner script like this can be most helpful when using a supercomputer cluster,
because it can send a series of individual job requests in to the cluster.

USAGE:
runner_power.sh <runMode>
or optional:
runner_power.sh <runMode> <paramsCONFIG>

  runMode:
    1: used to submit a large batch of calculations
    2: used to summarize the large batch when they are done. Mode 3 will automatically run afterwards.
    3: used to produce the final output after mode 2 has summarized data.

  paramsCONFIG (optional):
    0: DEFAULT. reads in the data with input path specified and formatted as described in ReadMe.md, default power calculation parameters.
    1: shortcut / saved parameter configuration for seizure tracking 'ST' dataset, as presented in the paper.
    2: shortcut / saved parameter configuration for covid hospitalization risk prediction 'COVA' dataset, as presented in the paper.
    3: shortcut / saved parameter configuration for brain age, longitudinal survival analysis datatset, as presented in the paper.

MODIFY THIS RUNNER SCRIPT FOR YOUR OWN USE."
}

HelpParameters()
{
echo "
In runner_power, user has to set the filepath to the dataset to be analyzed, and the output directory. Further, SSAML parameters can be specified:  
-- peopleTF: 1 if patient based, 0 if event based. (default 1)  
-- survivalTF: 1 for survival analysis/dataset, 0 if not (default 0).  
-- resampReps: number of repitions (outer loop) to be performed (default 1000).  
-- bootReps: bootstrap repitions in inner loop (default 40)  
"
}


if [ $# -eq 0 ]; then
Help
exit
fi

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         HelpParameters
         exit;;
   esac
done


# Input parameters
runMode=$1
# Optional input parameters: paramsCONFIG=$2 (see below).

re='^[0-9]+$'
if ! [[ $runMode =~ $re ]] ; then
   echo "ERROR. Argument 1 <runMode> needs to be an integer {1, 2, 3}." >&2; exit 1
fi
if (($runMode < 1 || $runMode > 3)); then
  echo "ERROR. Argument 1 <runMode> needs to be an integer {1, 2, 3}." >&2; exit 1
fi

# constants
p=`pwd`
PYTHON=python  # this depends on which command do you run python, usually it's `python` or `python3`

# if 1 here, then submit jobs to a supercomputer. If 0 here, run commands locally
use_supercomputer=0

## 
# Notes:
#  infile - where your input file will be found. Data format expected as described in data_format_preprocess.ipynb, in short:
#         a) for non-survival analysis: columns ['ID', 'event', 'p']
#         b) for survival analysis: columns ['ID', 'C', 'T', 'z'].
#  outdir - where output files should be saved
#  peopleTF - 1 if patient based, 0 if event based
#  survivalTF - 1 if survival analysis required, 0 if not
#  resampReps - number of repitions (outer loop) to be performed by power.py (as Figure 1, block C in paper)
#  bootReps - bootstrap repitions in inner loop (as Figure 1, block B in paper)
#  ilist - a list of what iteration numbers to run, where each iteration has resampReps iterations within (between 0 and 9999)
#  maxlist - a list of candidate number of patients/events to test for sample size power (any size ok)
#  conflist - a list of candidate confidence range (e.g. .95 = 95% confidence range) to test (any size ok)

case $# in
    1)
        # no paramsCONFIG specified, proceed with default value 0.
        paramsCONFIG=0
        ;;
    2)
        # paramsCONFIG argument passed.
        paramsCONFIG=$2
        ;;
    *)
        printf "%b" "ERROR. Maximum 2 arguments <runMode> <paramsCONFIG> are accepted."
        exit 1
        ;;
esac

case $paramsCONFIG in
# ***** YOU WILL DO YOUR EDITING HERE, MOST LIKELY FOR CASE 0
# CONVENIENCE shortcuts/savings of parameter configurations
# user with their own data can either use default power calculations parameters as set in case paramsCONFIG == 0 (and specify
# input and output paths), or add a paramsCONFIG shortcut (either modifyting case 1-3 or adding cases 4+).

    0)
        infile='SET_BY_USER'
        outdir='SET_BY_USER'
        peopleTF=1
        survivalTF=0
        resampReps=40
        bootReps=1000
        ilist=`seq 0 1 99`
        maxlist='500 1000 1500 2000'
        conflist='0.955 0.997 0.9999 0.999999'
        ;;
    1)
        # This is for a ST type dataset, with repeated measures in a time series
        # and multiple subjects
        infile='/home/wolfgang/repos/SSAML/sample_data_st.csv'
        outdir='/home/wolfgang/repos/SSAML/OUTst'
        peopleTF=1 # seizure tracking analysis is done on subject level
        survivalTF=0 # no survival analysis data
        resampReps=40
        bootReps=1000
        ilist=`seq 0 1 24`
        maxlist='20 40 60 80'
        conflist='0.955 0.997 0.9999 0.999999'
        ;;
    2)
        # This is for a COVA type dataset, with single measures for each patient
        # and multiple patients. No survival analysis here. The analysis is to
        # determine the number of EVENTS, not number of PATIENTS.
        infile='/home/wolfgang/repos/SSAML/sample_data_cova.csv'
        outdir='/home/wolfgang/repos/SSAML/OUTcovaFAKE'
        peopleTF=0 # covid risk analysis is done on event level
        survivalTF=0 # no survival analysis data
        resampReps=100
        bootReps=1000
        maxlist='40 50 60 70'
        ilist=`seq 0 1 10`
        #resampReps=40
        #maxlist='125 150 175 200'
        #ilist=`seq 0 1 24`
        conflist='0.955 0.997 0.9999 0.999999'
        ;;
    3)
        # This is for a BAI type dataset, with single measures for each patient
        # and a survival analysis conducted.
        infile='/home/wolfgang/repos/SSAML/sample_data_bai_mortality.csv'
        outdir='/home/wolfgang/repos/SSAML/OUTbai'
        peopleTF=1 # brain age mortality analysis is done on subject level
        survivalTF=1 # brain age mortality is a longitudinal, survival analysis dataset.
        resampReps=10
        bootReps=1000
        ilist=`seq 0 1 99`
        maxlist='500 1000 1500 2000'
        conflist='0.955 0.997 0.9999 0.999999'
        ;;
    *)
        echo "ERROR. Invalid paramsCONFIG specified."
        exit 1
        ;;
esac

if [ ! -f "$infile" ]; then
    echo "ERROR. Specified infile $infile does not exist."
    exit 1
fi

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
                    if [[ $use_supercomputer -eq 1 ]]
                    then
                        sbatch run_power.sh $runMode $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps $use_supercomputer $bootReps
                    else
                        echo $p/run_power.sh $runMode $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps $use_supercomputer $bootReps
                        $p/run_power.sh $runMode $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps $use_supercomputer $bootReps
                    fi
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
                echo "$PYTHON $p/power.py $runMode $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps $bootReps"
                $PYTHON $p/power.py $runMode $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps $bootReps
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
        $PYTHON $p/power.py $runMode $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps $bootReps
        ;;
    3)
        runMode=3
        cd $outdir
        maxPts=0
        confint=0
        $PYTHON $p/power.py $runMode $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps $bootReps
        ;;
    *)
        echo "ERROR. Only runMode 1,2,3 are allowed."
        ;;
esac

