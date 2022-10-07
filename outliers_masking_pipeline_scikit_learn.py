#! /usr/bin/env python3
"""
Code to mask outliers with a function, column transformer, and pipeline. It
also deals with outliers apart from masking, where X values >= 0.

The goal is to mask outliers and then process these within scikit-learn,
not within pandas, in order to avoid data leakage.

./outliers_masking_pipeline_scikit_learn.py >
    outliers_masking_pipeline_scikit_learn.txt

TODO:
- test with X1-X13
- how to handle outliers < 0
- how to handle y outliers
- delete diagnostic code
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
    for feature, lowvalue, highvalue in maskvalues:
        df[feature] = df[feature].mask(
            cond=(df[feature] <= lowvalue) | (df[feature] >= highvalue),
            other=pd.NA
        )
    return pd.DataFrame(data=df)


def main():
    global maskvalues
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
    features = [
        "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11",
        "X12", "X13", "X14"
    ]
    target = "Y"
    print("outliers_masking_pipeline_scikit_learn.py")
    print()
    df = ds.read_file(file_name=file_data, skip_blank_lines=False)
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
    mask_values = FunctionTransformer(mask_outliers)
    imputer = SimpleImputer()
    # imputer = KNNImputer(n_neighbors=10)
    imputer_pipeline = make_pipeline(mask_values, imputer)
    column_transformer = make_column_transformer(
        (imputer_pipeline, features),
        remainder="drop"
    )
    # column_transformer.fit_transform(df)
    # print("mask outliers in df:")
    # print(column_transformer.fit_transform(df))
    # print()
    linear_regression = LinearRegression(fit_intercept=True)
    linear_regression_selection = LinearRegression(fit_intercept=True)
    feature_selection = SelectFromModel(
        linear_regression_selection,
        threshold='median'
    )
    pipeline = make_pipeline(
        column_transformer,
        feature_selection,
        linear_regression
    )
    print(pipeline)
    print()
    pipeline.fit(X=X, y=y)
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
    print("best score:", grid_search.best_score_)
    print()
    print("best parameters:")
    print(grid_search.best_params_)
    print()
    # print("linear regression intercept:")
    # print(pipeline.fit(X=X, y=y).named_steps["linearregression"].intercept_)
    # print()
    # print("linear regression coefficients:")
    # print(pipeline.fit(X=X, y=y).named_steps["linearregression"].coef_)
    # print()
    # predictions_ndarray = pipeline.predict(X=X_new)
    print("linear regression intercept:")
    print(pipeline.named_steps.linearregression.intercept_.round(3))
    print()
    print("linear regression coefficients:")
    print(pipeline.named_steps.linearregression.coef_.round(3))
    print()
    print('nSelected features')
    print(X.columns[feature_selection.get_support()])
    print()
    predictions_ndarray = grid_search.predict(X=X_new)
    predictions_series = pd.Series(
        data=predictions_ndarray,
        index=X_new.index,
        name='Y predictions'
    )
    X_new_predictions = pd.concat(
        objs=[X_new, predictions_series],
        axis='columns'
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
