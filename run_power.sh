#!/bin/bash
#SBATCH -n 1                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -n, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 0-08:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=16G                           # Memory total in MB (for all cores)
#SBATCH -o slurm/st_%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e slurm/st_%j.err                 # File to which STDERR will be written, including job ID
#SBATCH --mail-type=FAIL                    # Type of email notification- BEGIN,END,FAIL,ALL


runMode=$1
iterNumber=$2
maxPts=$3
confint=$4
infile=$5
outdir=$6
peopleTF=$7
survivalTF=$8
resampReps=${9}
use_supercomputer=${10}
bootReps=${11}

if [[ $use_supercomputer -eq 1 ]]
then
    module load gcc/6.2.0
    module load python/3.7.4
    source ~/deep37/bin/activate

    cd /home/dmg16/SSAML
fi
python -u power.py $runMode $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps $bootReps
