for perc in 5
do

    infile=/data/interesting_side_projects/SSAML/github_fork/test_outlier/datasets/simulated_dataset_classfication_outlier_perc$perc.csv
    outfolder=/data/interesting_side_projects/SSAML/github_fork/test_outlier/result_outlier_perc$perc
    # run mode 1 and mode 2 with datatype=0 (custom)
    ./runner_power.sh 1 0 $infile $outfolder
    ./runner_power.sh 2 0 $infile $outfolder

done
