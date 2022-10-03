#! /usr/bin/env python3
"""
Example of scikit-learn for supervised machine learning, Xs are measurements
"""

from pathlib import Path
import time

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import datasense as ds
import pandas as pd


def main():
    features = [
        "X1", "X2", "X3", "X4", "X5", "X6", "X7",
        "X8", "X9", "X10", "X11", "X12", "X13", "X14"
    ]
    file_name = "lunch_and_learn.csv"
    target = "Y"
    print("example_machine_learning_barebones.py")
    print()
    df = ds.read_file(file_name=file_name)
    # scikit-learn does not support removing outlines within a Pipeline
# Set lower and upper values to remove outliers
    mask_values = [
        ("X1", -20, 20),
        ("X2", -25, 25),
        ("X3", -5, 5),
        ("X4", -10, 10),
        ("X5", -3, 3),
        ("X6", -5, 5),
        ("X7", -13, 13),
        ("X8", -9, 15),
        ("X9", -17, 15),
        ("X10", -16, 15),
        ("X11", -16, 17),
        ("X12", -16, 17),
        ("X13", -20, 23)
    ]
    ds.dataframe_info(
        df=df,
        file_in=file_name
    )
    # Replace outliers with NaN
    for column, lowvalue, highvalue in mask_values:
        df[column] = df[column].mask(
            cond=(df[column] <= lowvalue) | (df[column] >= highvalue),
            other=pd.NA
        )
    ds.dataframe_info(
        df=df,
        file_in=file_name
    )
    # delete empty rows for the target or impute missing values
    df = df.dropna(subset=[target])
    # df[target] = df[target].fillna(df[target].mean())
    X = df[features]
    y = df[target]
    # impute mean for missing target values or delete empty rows
    simple_imputer = SimpleImputer(missing_values=pd.NA)
    column_transformer = make_column_transformer(
        (simple_imputer, features),
        remainder="passthrough"
    )
    linear_regression= LinearRegression()
    pipeline = make_pipeline(column_transformer, linear_regression)
    pipeline.fit(X=X, y=y)
    print("Regression intercept")
    print(pipeline.named_steps.linearregression.intercept_.round(3))
    print("Regression coefficients")
    print(pipeline.named_steps.linearregression.coef_.round(3))
    print()


if __name__ == "__main__":
    main()
