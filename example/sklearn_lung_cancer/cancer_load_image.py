# import os
# import warnings
# import sys

# import pandas as pd
# import numpy as np

# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import ElasticNet

# from urllib.parse import urlparse
# import mlflow
# import mlflow.sklearn

# import logging

# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)

from sklearn.datasets import load_sample_images
dataset = load_sample_images()

print(len(dataset.images))

first_img_data = dataset.images[0]
print(first_img_data)

print(first_img_data.shape)
print(first_img_data.type)
