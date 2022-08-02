import pandas as pd
from sklearn.model_selection import train_test_split

# dataset = pd.read_csv(r"/mnt/e/MLOps/new_sklearn/Scikit-Learn-Study_MLOps/example/sklearn_lung_cancer/ThoraricSurgery.csv")
# Import csv files from `public/csv/` file-directory
dataset = pd.read_csv("../public/csv/ThoraricSurgery.csv")

# print('Dataset -> {0}'.format(dataset))

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(dataset)
# print("{0}".format(test))
# print("{0}".format(train.drop(["293"], axis=1)))
# print("{0}".format(train[["293"]]))


# The predicted column is "293" which is
train_x = train.drop(["293"], axis=1)
