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

module load gcc/6.2.0
module load python/3.7.4
source ~/deep37/bin/activate

cd /home/dmg16/SSAML

runMode=$1
dataTYPE=$2
iterNumber=$3
maxPts=$4
confint=$5
infile=$6
outdir=$7
peopleTF=$8
survivalTF=$9
resampReps=$10

python -u power.py $runMode $dataTYPE $iterNumber $maxPts $confint $infile $outdir $peopleTF $survivalTF $resampReps
