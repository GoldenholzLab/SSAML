cc=1
for Nfeat in 10 100; do
    for classratio in 1 10; do
        for flipy in 0.1 0.2; do
            echo
            echo ==================================================
            echo $cc, Nfeat=$Nfeat, classratio=$classratio, flipy=$flipy
            echo ==================================================
            echo
            infile=/data/HaoqiSun/SSAML/datasets/simulated_dataset_classfication_Nfeat${Nfeat}_classratio${classratio}_flipy${flipy}_randomseed2020.csv
            outfolder=/data/HaoqiSun/SSAML/github/simulation/OUTsimulation_Nfeat${Nfeat}_classratio${classratio}_flipy${flipy}_randomseed2020
            ./runner_power.sh 1 3 $infile $outfolder
            ./runner_power.sh 2 3 $infile $outfolder
            ((cc=cc+1))
        done
    done
done
