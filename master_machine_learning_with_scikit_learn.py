#! /usr/bin/env python3
"""
Master machine learning with scikit-learn
"""

from pathlib import Path
import time

import datasense as ds
import pandas as pd
import sklearn


def main():
    # Check installed versions of scikit-learn, pandas
    print("installed scikit-learn version:", sklearn.__version__)
    print("installed pandas version:      ", pd.__version__)
    # 2.1 Loading and exploring a dataset
    df = pd.read_csv("titanic_train.csv", nrows=10)
    print(df)
    # Use intuition to select two features
    # Create the X DataFrame
    X = df[["Parch", "Fare"]]
    print(X)
    # Create the y Series
    y = df["Survived"]
    print(y)
    # Check the shapes of X, y
    print(X.shape)
    print(y.shape)


if __name__ == "__main__":
    main()
