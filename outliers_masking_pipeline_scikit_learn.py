#! /usr/bin/env python3
"""
Code to mask outliers with a function, column transformer, and pipeline. It
also deals with outliers apart from masking, where X values >= 0.

The goal is to mask outliers and then process these within scikit-learn,
not within pandas, in order to avoid data leakage.

./outliers_masking_pipeline_scikit_learn.py >
    outliers_masking_pipeline_scikit_learn.txt

TODO:
- test with X1-X14
- how to handle outliers < 0
- try other imputers than SimpleImputer
- delete diagnostic code
"""

from pathlib import Path

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import datasense as ds
import pandas as pd


def mask_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mask outliers.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    maskvalues : List[Tuple[str, float, float]]
        The feature and mask values.

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
    maskvalues = [("X1", -4, 4)]
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
    linear_regression = LinearRegression()
    imputer = SimpleImputer()
    imputer_pipeline = make_pipeline(mask_values, imputer)
    column_transformer = make_column_transformer(
        (imputer_pipeline, features),
        remainder="drop"
    )
    column_transformer.fit_transform(df)
    # print("mask outliers in df:")
    # print(column_transformer.fit_transform(df))
    # print()
    pipeline = make_pipeline(column_transformer, linear_regression)
    pipeline.fit(X=X, y=y)
    print("linear regression intercept:")
    print(pipeline.fit(X=X, y=y).named_steps["linearregression"].intercept_)
    print()
    print("linear regression coefficients:")
    print(pipeline.fit(X=X, y=y).named_steps["linearregression"].coef_)
    print()
    predictions_ndarray = pipeline.predict(X=X_new)
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
