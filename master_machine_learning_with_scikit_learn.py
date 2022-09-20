#! /usr/bin/env python3
"""
Master machine learning with scikit-learn
"""

from pathlib import Path
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import datasense as ds
import pandas as pd
import sklearn


def main():
    print()
    print("installed scikit-learn version:", sklearn.__version__)
    print("installed pandas version:      ", pd.__version__)
    print()
    print("2.1 Loading and exploring a dataset")
    print()
    df = pd.read_csv(filepath_or_buffer="titanic_train.csv", nrows=10)
    print("First ten rows of titanic_train.csv")
    print()
    print(df)
    # Use intuition to select two features
    # Create the X DataFrame
    X = df[["Parch", "Fare"]]
    print()
    print("X DataFrame:")
    print(X)
    print()
    # Create the y Series
    y = df["Survived"]
    print("y Series:")
    print(y)
    print()
    # Check the shapes of X, y
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()
    print("2.2 Building and evaluate a model")
    print()
    logreg = LogisticRegression(solver="liblinear", random_state=1)
    crossvalscore = cross_val_score(
        estimator=logreg, X=X, y=y, cv=3, scoring="accuracy"
    ).mean()
    print("Cross-validation score:", crossvalscore)


if __name__ == "__main__":
    main()
