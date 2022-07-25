from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load Data
iris = load_iris()

# Initiate PCA and Classifier
pca = PCA()
classifier = DecisionTreeClassifier()

# transform / fit

X_transformed = pca.fit_transform(iris.data)
classifier.fit(X_transformed, iris.target)

# predict "new" data
# (I'm faking it here by using the original data)

newdata = iris.data

# transform new data using already fitted pca
# (don't re-fit the pca)

