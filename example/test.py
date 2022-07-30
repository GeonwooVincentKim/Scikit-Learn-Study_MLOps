# import os

# from random import random, randint

# from mlflow import log_mertric, log_param, log_artifacts

import mlflow

mlflow.start_run()
mlflow.log_param("my", "param")
mlflow.log_metric("score", 100)
mlflow.end_run()
