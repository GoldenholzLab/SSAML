#!/bin/bash

# constants
p=`pwd`
maxPts=1613
localMAC=0
if [[ $localMAC -eq 1 ]]; then
    infile='/myfiles/inputDataFile.csv'
    outdir='/myfiles/dir_for_outputs'
else
    infile='/home/dmg16/deepMan/OUTPUT_for_paper/holdoutisThisReal_v2.csv'
    outdir='/home/dmg16/SSAML/OUT'
fi
dataTYPE=0
runMode=$1

conflist='0.955 0.997 0.9999 0.999999'
ilist=`seq 0 1 249`
maxlist='100 150 200 250'
iterNumber=0
case $runMode in
    1)
        # This is runMode 1 - submit batches of scripts here (several hours)
        runMode=1
        for confint in $conflist; do
            for maxPts in $maxlist; do
                for iterNumber in $ilist; do            
                    sbatch run_power.sh $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF
                done
            done
        done
        ;;
    2)
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
                python $p/power.py $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF
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
        python $p/power.py $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF
        ;;
    *)
        ;;
esac




