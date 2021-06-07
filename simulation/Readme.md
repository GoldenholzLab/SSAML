## Simulation

Several simulated datasets are generated for
<!-- * classification vs. regression; -->
* different numbers of features;
<!-- * different sizes of the training set; -->
* different levels of class imbalance;<!--  \[classification only\]; -->
* different levels of class mislabeling.<!--  \[classification only\];-->
<!-- * different levels of feature colinearity \[regression only\]; -->
<!-- * different levels of added Gaussian noise \[regression only\]. -->

We assume we know
* the ground truth accuracy for classification = 1 - class mislabeling percent;
<!-- * the ground truth RMSE for regression = standard deviation of added Gaussian noise. -->

Each condition we generate 2 versions with different random seeds.
