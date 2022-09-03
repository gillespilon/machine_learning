#! /usr/bin/env python3
"""
Machine learning of the iris dataset.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn import metrics


def main():
    # load the data from a scikit-learn built-in dataset
    iris = load_iris()
    print("iris feature names")
    print(iris.feature_names)
    print("iris target names")
    print(iris.target_names)
    X = iris.data
    y = iris.target
    print("iris X data")
    print(X)
    print(X.shape)
    print("iris y data")
    print(y)
    print(y.shape)
    # use K-nearest neighbors
    print("K-neighbors classification, n_neighbors=1")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    result = knn.predict([[3, 5, 4, 2]])
    print(result)
    X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
    result = knn.predict(X_new)
    print(result)
    print("K-neighbors classification, n_neighbors=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    result = knn.predict(X_new)
    print(result)
    # use logistic regression
    print("logistic regression")
    logreg = LogisticRegression()
    logreg.fit(X, y)
    result = logreg.predict(X_new)
    print(result)
    # train the model with all of the data, evaluate accuracy
    print("training accuracy on all data for logistic regression")
    logreg = LogisticRegression()
    logreg.fit(X, y)
    y_pred = logreg.predict(X)
    print(metrics.balanced_accuracy_score(y, y_pred))
    # do the same for knn with k=5
    print("training accuracy on all data for knn=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    print(metrics.balanced_accuracy_score(y, y_pred))
    # do the same for knn with k=1
    print("training accuracy on all data for knn=r")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    print(metrics.balanced_accuracy_score(y, y_pred))
    # train-test-split, logistic regression, teseting accuracy
    print("train-test-split, testing accuracy, logistic regression")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=4
    )
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(metrics.balanced_accuracy_score(y_test, y_pred))
    # train-test-split, knn with k=5
    print("train-teset-split, testing accuracy, knn=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(metrics.balanced_accuracy_score(y_test, y_pred))
    # train-test-split, knn with k=1
    print("train-teset-split, testing accuracy, knn=1")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(metrics.balanced_accuracy_score(y_test, y_pred))
    # predict out-of-sample data with best model
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X, y)   # use all of the data
    y_pred = knn.predict([[3, 5, 4, 2]])
    print(y_pred)


if __name__ == "__main__":
    main()
