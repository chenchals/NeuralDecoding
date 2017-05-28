# Download IDE and code

1. Download Pycharm @ https://www.jetbrains.com/pycharm/
2. Install Pycharm
3. Open Pycharm
4. Click VCS->Git->Clone
5. Copy and paste this url in `Git Repository URL` box https://github.com/colpain/NeuralDecoding.git
6. Click Clone


# How to use

## Experiment Config
Use `experiment_config.json` config file to config the experiment.

Explanations of config file fields are given below:

- model_name [REQUIRED]: the name of the model used in the pipeline, available models are:
  1. "NearestNeighbors" 
  2. "RBF SVM"
  3. "DecisionTree"
  4. "RandomForest"
  5. "AdaBoost"
  6. "KNN"
  7. "RadiusNeighbors"
- hyper_parameter [REQUIRED]: the hyper parameter to define the model complexity, the higher the more complex model is
- mode: experiment mode [OPTIONAL]: available options are [100, 1], means either run 100 calls per time point of 1 call per time point, more options will be added in future. The default value of this field is 1