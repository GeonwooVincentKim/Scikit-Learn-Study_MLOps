import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

# dataset = pd.read_csv(r"/mnt/e/MLOps/new_sklearn/Scikit-Learn-Study_MLOps/example/sklearn_lung_cancer/ThoraricSurgery.csv")
# Import csv files from `public/csv/` file-directory
dataset = pd.read_csv("../public/csv/ThoraricSurgery.csv")

# print('Dataset -> {0}'.format(dataset))

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(dataset)
# print("{0}".format(test))
# print("{0}".format(train.drop(["293"], axis=1)))
# print("{0}".format(train[["293"]]))


# The predicted column is "293" which is a scalar from [3, 9]
train_x = train.drop(["293"], axis=1)
test_x = test.drop(["293"], axis=1)
train_y = train[["293"]]
test_y = test[["293"]]

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)

predicted_qualities = lr.predict(test_x)
