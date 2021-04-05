#!/bin/bash

# constants
peopleTF=1
maxPts=1613
survivalTF=0
infile=/myfiles/inputDataFile.csv
outdir=/myfiles/dir_for_outputs

# submit 100 iters
runMode=$1

#runMode=0
#sbatch run_bai 0 0 0 0
# this determined that .99 is good for confint
conflist='0.955 0.997 0.9999 0.999999'
ilist=`seq 1 1 9999`
maxlist='125 150 175 200'
iterNumber=0
case $runMode in
    1)
        # This is runMode 1 - submit batches of scripts here (several hours)
        for confint in $conflist; do
            for maxPts in $maxlist; do
                #for iterNumber in $ilist; do
                runMode=1
                python power.py $runMode $peopleTF $iterNumber $maxPts $confint $survivalTF $infile $outdir
                #done
                maxP=`printf "%04d" $maxPts`
                FILE=num${maxP}_${confint}.csv
                if [ ! -f "$FILE" ]; then
                    cat num${maxP}????_${confint}.csv > $FILE
                    mv num${maxP}????_${confint}.csv holder/
                fi
                runMode=2
                python power.py $runMode $peopleTF $iterNumber $maxPts $confint $survivalTF $infile $outdir
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
        runMode=3
        python power.py $runMode $peopleTF $iterNumber $maxPts $confint $survivalTF $infile $outdir
        ;;
    *)
        ;;
esac




