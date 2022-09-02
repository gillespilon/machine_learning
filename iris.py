#! /usr/bin/env python3
"""
Machine learning of the iris dataset.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    print(type(iris))
    print(iris.data)
    print(iris.feature_names)
    print(iris.target)
    print(iris.target_names)
    print(iris.data.shape)
    print(iris.target.shape)
    X = iris.data
    y = iris.target
    print(X)
    print(y)
    print(X.shape)
    print(y.shape)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    result = knn.predict([[3, 5, 4, 2]])
    print(result)
    X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
    result = knn.predict(X_new)
    print(result)

if __name__ == "__main__":
    main()
