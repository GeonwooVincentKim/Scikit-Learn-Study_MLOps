import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse

import mlflow
import mlflow.sklearn


# Evaluate metrics
def eval_metrics(actual, pred):
	rmse = np.sqrt(mean_squared_error(actual, pred))
	mae = mean_absolute_error(actual, pred)
	r2 = r2_score(actual, pred)

	return rmse, mae, r2


# dataset = pd.read_csv(r"/mnt/e/MLOps/new_sklearn/Scikit-Learn-Study_MLOps/example/sklearn_lung_cancer/ThoraricSurgery.csv")
# Import csv files from `public/csv/` file-directory
# dataset = pd.read_csv("../public/csv/ThoraricSurgery.csv", sep=';')
dataset = pd.read_csv("../public/csv/ThoraricSurgery.csv")

print('Dataset -> {0}'.format(dataset))
# print("Dataset 293 -> {0}".format(dataset.toList()))

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(dataset)
print("Test -> \n{0}".format(test))
print("\n\nTrain -> \n{0}".format(train))
# print("{0}".format(train.drop(["293"], axis=1)))
print("{0}".format(train))


# The predicted column is "293" which i s a scalar from [3, 9]
train_x = train.drop(["293"], axis=1)
test_x = test.drop(["293"], axis=1)
train_y = train[["293"]]
test_y = test[["293"]]

print("Train X -> {0}".format(train_x))

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


# Run ElasticNet
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)

predicted_qualities = lr.predict(test_x)
print(predicted_qualities)

(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

# # Print out ElasticNet Model Metrics
print("Elasticnet Model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)

# # Log mlflow attributes for mlflow UI
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)
# mlflow.sklearn.log_model(lr, "model")

tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
# print(tracking_url_type_store)

# if tracking_url_type_store != "file":
# 	mlflow.sklearn.log_model(lr, "model", registered_model_name="LungCancerModel")

# else:
# 	mlflow.sklearn.log_model(lr, "model")