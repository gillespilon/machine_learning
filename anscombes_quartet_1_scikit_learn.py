#! /usr/bin/env python3
"""
Example of simple linear regression using scikit-learn

./anscombes_quartet_1.py
"""

from pathlib import Path

from sklearn.linear_model import LinearRegression
import datasense as ds
import pandas as pd


def main():
    FILE_NAME = Path("anscombes_quartet_1.csv")
    FNAME = Path("anscombes_quartet_1.svg")
    FEATURE = "x"
    FORMAT = "svg"
    TARGET = "y"
    df = ds.read_file(file_name=FILE_NAME)
    # X must be two-dimensional, such as a pandas DataFrame
    X = df[[FEATURE]]
    # y must be one-dimensional, such as a pandas Series
    y = df[TARGET]
    model = LinearRegression()
    model = model.fit(X=X, y=y)
    # convert numpy ndarray to pandas Series
    predictions = pd.Series(data=model.predict(X=X))
    intercept = model.intercept_
    coefficients = model.coef_
    print("intercept  ", intercept.round(decimals=3))
    print("coefficient", coefficients.round(decimals=3))
    # X must be a Series
    fig, ax = ds.plot_scatter_line_x_y1_y2(
        X=X.squeeze(),
        y1=y,
        y2=predictions,
        figsize=(8, 6)
    )
    ds.despine(ax=ax)
    fig.savefig(
        fname=FNAME,
        format=FORMAT
    )


if __name__ == "__main__":
    main()
