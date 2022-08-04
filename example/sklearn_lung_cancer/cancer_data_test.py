import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

# dataset = pd.read_csv(r"/mnt/e/MLOps/new_sklearn/Scikit-Learn-Study_MLOps/example/sklearn_lung_cancer/ThoraricSurgery.csv")
# Import csv files from `public/csv/` file-directory
# dataset = pd.read_csv("../public/csv/ThoraricSurgery.csv", sep=';')
dataset = pd.read_csv("../public/csv/ThoraricSurgery.csv")

print('Dataset -> \n{0}'.format(dataset))
# print("Dataset 293 -> {0}".format(dataset.toList()))

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(dataset)
print("Test -> \n{0}".format(test))
print("\n\nTrain -> \n{0}".format(train))
# print("{0}".format(train.drop(["293"], axis=1)))
print("{0}".format(train))
