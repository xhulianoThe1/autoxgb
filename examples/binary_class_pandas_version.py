from src.autoxgb import AutoXGB

import pandas as pd


# required parameters:
train_filename = pd.read_csv("../data_samples/binary_classification.csv")

output = "output3"

# optional parameters
test_filename = None
task = None
idx = None
targets = ["income"]
features = None
categorical_features = None
use_gpu = False
num_folds = 5
seed = 42
num_trials = 100
time_limit = 360
fast = False

# Now its time to train the model!
axgb = AutoXGB(
    train_filename=train_filename,
    output=output,
    test_filename=test_filename,
    task=task,
    idx=idx,
    targets=targets,
    features=features,
    categorical_features=categorical_features,
    use_gpu=use_gpu,
    num_folds=num_folds,
    seed=seed,
    num_trials=num_trials,
    time_limit=time_limit,
    fast=fast,
)
axgb.train()
