#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Machine learning with text in PYthon."
"""

from pathlib import Path
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import datasense as ds
import pandas as pd


def main():
    # Model-building in scikit-learn (refresher)
    # load the iris data as an example
    iris = load_iris()
    # store the feature matrix X and the response vector y
    X = iris.data
    y = iris.target
    # check the shapes of X and y
    print("shape of X")
    print(X.shape)
    print("shape of y")
    print(y.shape)
    print("X object type")
    print(type(X))
    print("y object type")
    print(type(y))
    # examine the first five rows of X
    print("First five rows of X:")
    print(pd.DataFrame(data=X, columns=iris.feature_names).head())
    # the features must be numeric
    # instantiate the model with the default parameters
    knn = KNeighborsClassifier()
    # fit the model with the data; occurs in-place
    knn.fit(X=X, y=y)
    # predict the response for a new observation
    print("prediction for 3, 5, 4, 2")
    print(knn.predict(X=[[3, 5, 4, 2]]))
    # Representing text as numerical data
    # example text for numerical training (SMS messages)
    simple_train = [
        "call you tonight",
        "Call me a cab",
        "please call me... PLEASE!",
    ]
    # instantiate CountVectorizer() with the default parameters
    vect = CountVectorizer()
    # learn the vocabulary of the training data; occurs in-place
    vect.fit(raw_documents=simple_train)
    # examine the fitted vocabulary
    print("Fitted vocabulary")
    print(vect.get_feature_names_out())
    # transform training data into a document-term matrix
    simple_train_dtm = vect.transform(raw_documents=simple_train)
    print("DTM sparse matrix")
    print(simple_train_dtm)
    # convert sparse matrix to dense matrix; does not change simple_train_dtm
    # the toarray() method is only so that we can "see" the result
    simple_train_dtm.toarray()
    print("Dense matrix")
    print(simple_train_dtm.toarray())
    print(
        pd.DataFrame(
            data=simple_train_dtm.toarray(),
            columns=vect.get_feature_names_out(),
        )
    )
    # example text for model testing
    simple_test = ["please don't call me"]
    # in order to make a prediction, the new observation must have the same
    # features as the training observations, both in number and meaning
    # transform testing data into a DTM sparse matrix using existing vocabulary
    simple_test_dtm = vect.transform(raw_documents=simple_test)
    # summary
    # vect.fit(train) learns the vocabulary of the training data
    # vect.transform(train) uses the fitted vocabulary to build a DTM
    # from the training data
    # vect.transform (test) uses the fitted voacbulary to build a DTM
    # from the testing data and ignores tokens it had not seen before
    # Reading the SMS data
    # read file into pandas from a URL
    path = Path("ML-text-main/data/sms.tsv")
    sms = pd.read_table(
        filepath_or_buffer=path, header=None, names=["label", "message"]
    )
    print("shape of sms")
    print(sms.shape)
    print("look at first ten rows")
    print(sms.head(10))
    print("Examine the class distribution")
    print(sms["label"].value_counts())
    # convert label to a numeric variable
    sms["label_num"] = sms["label"].replace(
        to_replace=["ham", "spam"], value=[0, 1]
    )
    # usual way to define X and y for use with a model
    # X = iris.data
    # y = iris.target
    # required way to define X and y with a CountVectorizer
    # X and y are one-dimensional
    # X is defined as one-dimensional because it will get converted by
    # CountVectorizer into a two-dimensional X
    X = sms["message"]
    y = sms["label_num"]
    # split X and y into training and test data sets
    # this must be done before vectorization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.25
    )
    print("X_train shape")
    print(X_train.shape)
    print("X_test shape")
    print(X_test.shape)
    # Vectorizing the SMS data
    # instantiate the vectorizer
    vect = CountVectorizer()
    # learn training data vocabulary, then use it to create a sparse DTM
    # the following two lines are as above
    # vect.fit(raw_documents=X_train)
    # X_train_dtm = vect.transform(raw_documents=X_train)
    # alternatively you can do this in one line
    X_train_dtm = vect.fit_transform(raw_documents=X_train)
    # you do not use a fit step on the test data, only on the training data
    # transform test data using fitted vocabulary into a sparse matrix DTM
    X_test_dtm = vect.transform(raw_documents=X_test)
    # Building a Naive Bayes model


if __name__ == "__main__":
    main()
