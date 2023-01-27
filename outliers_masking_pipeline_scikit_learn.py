#! /usr/bin/env python3
"""
Mask outliers with a function, column transformer, and pipeline, in order to
avoid data leakage.

In this example I use a simulated data set that has seven features (X1-X7)
that affect the target (Y) and six features (X8-X14) that have zero effect
on the target.

TODO:
- how to handle y outliers
"""

from pathlib import Path
import time

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
    Mask outliers

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    df : pd.DataFrame
        The output DataFrame.
    """
    for column, lowvalue, highvalue in MASK_VALUES:
        df[column] = df[column].mask(
            cond=(df[column] <= lowvalue) | (df[column] >= highvalue),
            other=pd.NA
        )
    return pd.DataFrame(data=df)


def main():
    start_time = time.perf_counter()
    global MASK_VALUES
    FEATURES = [
        "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11",
        "X12", "X13", "X14"
    ]
    FILE_PREDICTIONS = Path("outliers_missing_predictions.csv")
    OUTPUT_URL = "outliers_masking_pipeline_scikit_learn.html"
    FILE_NEW = Path("outliers_missing_new.csv")
    FILE_DATA = Path("outliers_missing.csv")
    HEADER_TITle = "Masking outliers"
    HEADER_ID = "Masking outliers"
    # there is no mask for X14 because the number of NaN is too large
    MASK_VALUES = [
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
    TARGET = "Y"
    original_stdout = ds.html_begin(
        output_url=OUTPUT_URL,
        header_title=HEADER_TITle,
        header_id=HEADER_ID
    )
    ds.script_summary(
        script_path=Path(__file__),
        action="started at"
    )
    print("outliers_masking_pipeline_scikit_learn.py")
    print()
    df = ds.read_file(
        file_name=FILE_DATA,
        skip_blank_lines=False
    )
    df = ds.optimize_columns(df=df)
    # df = pd.read_csv(filepath_or_buffer=FILE_DATA, skip_blank_lines=False)
    ds.dataframe_info(
        df=df,
        file_in=FILE_DATA
    )
    # delete empty rows for the TARGET or impute missing values
    df = ds.delete_empty_rows(
        df=df,
        list_columns=[TARGET]
    )
    ds.dataframe_info(
        df=df,
        file_in=FILE_DATA
    )
    # df[TARGET] = df[TARGET].fillna(df[TARGET].mean())
    # X is two-dimensional
    X = df[FEATURES]
    # y is one-dimensional
    y = df[TARGET]
    df_new = ds.read_file(
        file_name=FILE_NEW,
        skip_blank_lines=False
    )
    df_new = ds.optimize_columns(df=df_new)
    # df_new = pd.read_csv(filepath_or_buffer=FILE_NEW, skip_blank_lines=False)
    ds.dataframe_info(
        df=df_new,
        file_in=FILE_NEW
    )
    # X_new is two-dimensional
    X_new = df_new[FEATURES]
    mask = FunctionTransformer(func=mask_outliers)
    imputer = SimpleImputer()
    # imputer = KNNImputer(n_neighbors=10)
    imputer_pipeline = make_pipeline(mask, imputer)
    column_transformer = make_column_transformer(
        (imputer_pipeline, FEATURES),
        remainder="drop"
    )
    linear_regression = LinearRegression(fit_intercept=True)
    linear_regression_selection = LinearRegression(fit_intercept=True)
    feature_selection = SelectFromModel(
        estimator=linear_regression_selection,
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
    selected_coefficients = (
        pipeline.named_steps.linearregression.coef_.round(6)
    )
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
    #     path_or_buf=FILE_PREDICTIONS
    # )
    ds.save_file(
        df=X_new_predictions,
        file_name=FILE_PREDICTIONS
    )
    stop_time = time.perf_counter()
    ds.script_summary(
        script_path=Path(__file__),
        action="finished at"
    )
    ds.report_summary(
        start_time=start_time,
        stop_time=stop_time
    )
    ds.html_end(
        original_stdout=original_stdout,
        output_url=OUTPUT_URL
    )


if __name__ == "__main__":
    main()
