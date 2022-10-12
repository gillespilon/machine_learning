#! /usr/bin/env python3
"""
Mask outliers with a function, column transformer, and pipeline, in order to
avoid data leakage.

In this example I use a simulated data set that has seven features (X1-X7)
that affect the target (Y) and six features (X8-X14) that have zero effect
on the target.

./outliers_masking_pipeline_scikit_learn.py >
    outliers_masking_pipeline_scikit_learn.txt

TODO:
- how to handle y outliers
"""

from pathlib import Path

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import datasense as ds
import pandas as pd
import numpy as np


def mask_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mask outliers.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    df : pd.DataFrame
        The output DataFrame.
    """
    for column, lowvalue, highvalue in maskvalues:
        df[column] = df[column].mask(
            cond=(df[column] <= lowvalue) | (df[column] >= highvalue),
            other=pd.NA
        )
    return pd.DataFrame(data=df)


def main():
    global maskvalues
    features = [
        "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11",
        "X12", "X13", "X14"
    ]
    file_predictions = Path("outliers_missing_predictions.csv")
    file_new = Path("outliers_missing_new.csv")
    file_data = Path("outliers_missing.csv")
    # there is no mask for X14 because the number of NaN is too large
    maskvalues = [
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
    target = "Y"
    print("outliers_masking_pipeline_scikit_learn.py")
    print()
    df = ds.read_file(
        file_name=file_data,
        skip_blank_lines=False
    )
    # df = pd.read_csv(filepath_or_buffer=file_data, skip_blank_lines=False)
    ds.dataframe_info(
        df=df,
        file_in=file_data
    )
    # delete empty rows for the target or impute missing values
    df = df.dropna(subset=[target])
    # df[target] = df[target].fillna(df[target].mean())
    X = df[features]
    y = df[target]
    df_new = ds.read_file(file_name=file_new, skip_blank_lines=False)
    # df_new = pd.read_csv(filepath_or_buffer=file_new, skip_blank_lines=False)
    ds.dataframe_info(
        df=df_new,
        file_in=file_new
    )
    X_new = df_new[features]
    mask = FunctionTransformer(mask_outliers)
    imputer = SimpleImputer()
    # imputer = KNNImputer(n_neighbors=10)
    imputer_pipeline = make_pipeline(mask, imputer)
    column_transformer = make_column_transformer(
        (imputer_pipeline, features),
        remainder="drop"
    )
    linear_regression = LinearRegression(fit_intercept=True)
    linear_regression_selection = LinearRegression(fit_intercept=True)
    feature_selection = SelectFromModel(
        estimate=linear_regression_selection,
        threshold="median"
    )
    pipeline = make_pipeline(
        column_transformer,
        feature_selection,
        linear_regression
    )
    print(pipeline)
    print()
    pipeline.fit(
        X=X,
        y=y
    )
    print()
    hyper_parameters = {}
    hyper_parameters[
        "columntransformer__pipeline__simpleimputer__strategy"
    ] = ["mean", "median", "most_frequent", "constant"]
    hyper_parameters["selectfrommodel__threshold"] = [None, "mean", "median"]
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=hyper_parameters,
        cv=10
    )
    grid_search.fit(X=X, y=y)
    print("best hyperparameters:")
    print(grid_search.best_params_)
    print()
    print("best score:", grid_search.best_score_.round(6))
    print()
    print(
        "linear regression intercept:",
        pipeline.named_steps.linearregression.intercept_.round(6)
    )
    print()
    print("linear regression coefficients:")
    selected_features = X.columns[feature_selection.get_support()].to_list()
    selected_coefficients = pipeline.named_steps.linearregression.coef_.round(6)
    selected_importances = np.abs(
        pipeline.named_steps.selectfrommodel.estimator_.coef_[
            feature_selection.get_support()
        ]
    ).tolist()
    print(pd.DataFrame(
        list(zip(
            selected_features, selected_importances, selected_coefficients)),
        columns=["Features", "Importance", "Coefficients"]
    ))
    print()
    predictions_ndarray = grid_search.predict(X=X_new)
    predictions_series = pd.Series(
        data=predictions_ndarray,
        index=X_new.index,
        name="Y predictions"
    )
    X_new_predictions = pd.concat(
        objs=[X_new, predictions_series],
        axis="columns"
    )
    # X_new_predictions.to_csv(
    #     path_or_buf=file_predictions
    # )
    ds.save_file(
        df=X_new_predictions,
        file_name=file_predictions
    )


if __name__ == "__main__":
    main()
