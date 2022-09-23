#! /usr/bin/env python3
"""
Master machine learning with scikit-learn
"""

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import datasense as ds
import pandas as pd


def main():
    print("master_machine_learning_with_scikit_learn.py")
    print()
    features = ["Parch", "Fare", "Embarked", "Sex"]
    target = "Survived"
    df = ds.read_file(
        file_name="titanic_train.csv",
        nrows=10
    )
    X = df[features]
    y = df[target]
    df_new = ds.read_file(
        file_name="titanic_new.csv",
        nrows=10
    )
    X_new = df_new[features]
    one_hot_encoder = OneHotEncoder()
    column_transformer = make_column_transformer(
        (one_hot_encoder, ["Embarked", "Sex"]),
        ("passthrough", ["Parch", "Fare"])
    )
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X, y=y)
    pipeline.predict(X=X_new)
    print("pipeline.predict(X=X_new):", pipeline.predict(X=X_new))
    print()


if __name__ == "__main__":
    main()
