infile=/data/interesting_side_projects/SSAML/github_fork/test_skewed_distribution/datasets/simulated_dataset_classfication_skewed.csv
outfolder=/data/interesting_side_projects/SSAML/github_fork/test_skewed_distribution/result_skewed
# run mode 1 and mode 2 with datatype=0 (custom)
./runner_power.sh 1 0 $infile $outfolder
./runner_power.sh 2 0 $infile $outfolder
