#! /usr/bin/env python3
"""
Machine learning of the iris dataset.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import metrics
import datasense as ds
import seaborn as sns
import pandas as pd


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
    # train the model with all of the data, evaluate accuracy
    print("training accuracy on all data for logistic regression")
    logreg = LogisticRegression()
    logreg.fit(X=X, y=y)
    y_predicted = logreg.predict(X=X)
    print(metrics.balanced_accuracy_score(y_true=y, y_pred=y_predicted))
    # do the same for knn with k=5
    print("training accuracy on all data for knn=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X=X, y=y)
    y_predicted = knn.predict(X=X)
    print(metrics.balanced_accuracy_score(y_true=y, y_pred=y_predicted))
    # do the same for knn with k=1
    print("training accuracy on all data for knn=r")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X=X, y=y)
    y_predicted = knn.predict(X=X)
    print(metrics.balanced_accuracy_score(y_true=y, y_pred=y_predicted))
    # train-test-split, logistic regression, teseting accuracy
    print("train-test-split, testing accuracy, logistic regression")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=4
    )
    logreg = LogisticRegression()
    logreg.fit(X=X_train, y=y_train)
    y_predicted = logreg.predict(X=X_test)
    print(metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_predicted))
    # train-test-split, knn with k=5
    print("train-teset-split, testing accuracy, knn=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X=X_train, y=y_train)
    y_predicted = knn.predict(X=X_test)
    print(metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_predicted))
    # train-test-split, knn with k=1
    print("train-teset-split, testing accuracy, knn=1")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X=X_train, y=y_train)
    y_predicted = knn.predict(X=X_test)
    print(metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_predicted))
    # predict out-of-sample data with best model
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X=X, y=y)   # use all of the data
    y_predicted = knn.predict(X=[[3, 5, 4, 2]])
    print(y_predicted)
    # linear regression
    data_path = "iris_data.csv"
    data = pd.read_csv(
        filepath_or_buffer=data_path,
        index_col=0
    )
    print(data.head())
    # ds.dataframe_info(df=data, file_in=data_path)
    # using seaborn
    sns.pairplot(
        data,
        x_vars=['TV', 'Radio', 'Newspaper'],
        y_vars='Sales',
        height=7,
        kind='reg'
    )
    plt.show(block=True)
    # using datasense
    # for item in ["TV", "Radio", "Newspaper"]:
    #     X = data[item]
    #     y = data["Sales"]
    #     fig, ax = ds.plot_scatter_x_y(X=X, y=y)
    #     ax.set_ylabel(ylabel="Sales")
    #     ax.set_xlabel(xlabel=item)
    #     fig.savefig(fname=f"scatter_{item}_vs_Sales.png")

if __name__ == "__main__":
    main()
