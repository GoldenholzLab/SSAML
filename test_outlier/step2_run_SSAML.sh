infile=sample_data_bai_mortality_outlier_perc0.csv
outfolder=result_bai_mortality_outlier_perc0
# run mode 1 and mode 2 with datatype=0 (custom)
../runner_power.sh 1 0 $infile $outfolder
../runner_power.sh 2 0 $infile $outfolder
