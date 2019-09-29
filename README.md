# Pipeline

## How to run training

First of all, create a config. You may find some examples of configs in folders mnist_pipeline, cifar_pipeline and imagenet_pipeline.
Then, call:

`PYTHONPATH=. python3 bin/train.py path_to_config`

For example we want train cf_models

`PYTHONPATH=. python bin/train_collaborative_model.py configuration/base.py`