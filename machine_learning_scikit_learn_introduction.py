#! /usr/bin/env python3
"""
Machine learning of the iris dataset.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import metrics
import datasense as ds
import seaborn as sns
import pandas as pd
import numpy as np


def main():
    # Lesson 03
    print("Lesson 03")
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
    # Lesson 04
    print("Lesson 04")
    # use K-nearest neighbors
    print("K-neighbors classification, n_neighbors=1")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X=X, y=y)
    result = knn.predict(X=[[3, 5, 4, 2]])
    print(result)
    X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
    result = knn.predict(X=X_new)
    print(result)
    print("K-neighbors classification, n_neighbors=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X=X, y=y)
    result = knn.predict(X=X_new)
    print(result)
    # use logistic regression
    print("logistic regression")
    logreg = LogisticRegression()
    logreg.fit(X=X, y=y)
    result = logreg.predict(X=X_new)
    print(result)
    # Lesson 05
    print("Lesson 05")
    # train the model with all of the data, evaluate accuracy
    print("training accuracy on all data for logistic regression")
    logreg = LogisticRegression()
    logreg.fit(X=X, y=y)
    y_predicted = logreg.predict(X=X)
    print(metrics.accuracy_score(y_true=y, y_pred=y_predicted))
    # do the same for knn with k=5
    print("training accuracy on all data for knn=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X=X, y=y)
    y_predicted = knn.predict(X=X)
    print(metrics.accuracy_score(y_true=y, y_pred=y_predicted))
    # do the same for knn with k=1
    print("training accuracy on all data for knn=r")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X=X, y=y)
    y_predicted = knn.predict(X=X)
    print(metrics.accuracy_score(y_true=y, y_pred=y_predicted))
    # train-test-split, logistic regression, teseting accuracy
    print("train-test-split, testing accuracy, logistic regression")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=4
    )
    logreg = LogisticRegression()
    logreg.fit(X=X_train, y=y_train)
    y_predicted = logreg.predict(X=X_test)
    print(metrics.accuracy_score(y_true=y_test, y_pred=y_predicted))
    # train-test-split, knn with k=5
    print("train-teset-split, testing accuracy, knn=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X=X_train, y=y_train)
    y_predicted = knn.predict(X=X_test)
    print(metrics.accuracy_score(y_true=y_test, y_pred=y_predicted))
    # train-test-split, knn with k=1
    print("train-teset-split, testing accuracy, knn=1")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X=X_train, y=y_train)
    y_predicted = knn.predict(X=X_test)
    print(metrics.accuracy_score(y_true=y_test, y_pred=y_predicted))
    # predict out-of-sample data with best model
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X=X, y=y)   # use all of the data
    y_predicted = knn.predict(X=[[3, 5, 4, 2]])
    print(y_predicted)
    # Lesson 06
    print("Lesson 06")
    # linear regression
    data_path = "advertising.csv"
    data = pd.read_csv(
        filepath_or_buffer=data_path,
        index_col=0
    )
    print(data.head())
    # ds.dataframe_info(df=data, file_in=data_path)
    # using seaborn
    # sns.pairplot(
    #     data,
    #     x_vars=['TV', 'Radio', 'Newspaper'],
    #     y_vars='Sales',
    #     height=7,
    #     kind='reg'
    # )
    plt.show(block=True)
    # using datasense
    for item in ["TV", "Radio", "Newspaper"]:
        X = data[item]
        y = data["Sales"]
        fig, ax = ds.plot_scatter_x_y(X=X, y=y)
        ax.set_ylabel(ylabel="Sales")
        ax.set_xlabel(xlabel=item)
        fig.savefig(fname=f"scatter_{item}_vs_Sales.svg")
    feature_columns = ["TV", "Radio", "Newspaper"]
    # fit a model with three features
    X = data[feature_columns]
    response_column = "Sales"
    y = data[response_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X=X_train, y=y_train)
    print(linreg.intercept_)
    print(linreg.coef_)
    print(linreg.intercept_, list(zip(feature_columns, linreg.coef_)))
    y_predicted = linreg.predict(X=X_test)
    print(
        np.sqrt(metrics.mean_squared_error(y_true=y_test, y_pred=y_predicted))
    )
    # fit a model with two features
    feature_columns = ["TV", "Radio"]
    X = data[feature_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg.fit(X=X_train, y=y_train)
    y_predicted = linreg.predict(X=X_test)
    print(
        np.sqrt(metrics.mean_squared_error(y_true=y_test, y_pred=y_predicted))
    )
    # Lesson 07
    print("Lesson 07")
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=4
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X=X_train, y=y_train)
    y_predicted = knn.predict(X=X_test)
    print(
        "Accuracy score with random state = 4:",
        metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=3
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X=X_train, y=y_train)
    y_predicted = knn.predict(X=X_test)
    print(
        "Accuracy score with random state = 3:",
        metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=2
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X=X_train, y=y_train)
    y_predicted = knn.predict(X=X_test)
    print(
        "Accuracy score with random state = 2:",
        metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    # cv is the K in K-fold cross-validation
    scores = cross_val_score(
        estimator=knn, X=X, y=y, scoring="accuracy", cv=10
    )
    print("Mean of 10-fold cross-validation scores:", scores.mean())
    # search for optimate value of k for KNN
    k_range = list(range(1, 31))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(
            estimator=knn, X=X, y=y, cv=10, scoring="accuracy"
        )
        k_scores.append(scores.mean())
    print("k_scores:", k_scores)
    print("k_range:", k_range)
    fig, ax = ds.plot_scatter_x_y(X=pd.Series(k_range), y=pd.Series(k_scores))
    ax.set_ylabel(ylabel="Value of k for KNN")
    ax.set_xlabel(xlabel="Cross-validated accuracy")
    fig.savefig(fname="scatter_k_range_vs_k_scores.svg")


if __name__ == "__main__":
    main()
