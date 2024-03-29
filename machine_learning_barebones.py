#! /usr/bin/env python3
"""
Example of scikit-learn for supervised machine learning

- Xs are measurements
- Ys are measurements
- Process outliers and missing X values within scikit-learn
- Tune hyperparameters

Requires datasense: https://github.com/gillespilon/datasense
"""

from typing import List, Tuple
from pathlib import Path

from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import datasense as ds
import pandas as pd
import numpy as np


def mask_outliers(
    df: pd.DataFrame,
    mask: List[Tuple[str, float, float]]
) -> pd.DataFrame:
    """
    Mask outliers within a scikit-learn pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    mask : List[Tuple[str, float, float]]
        The list of mask values.

    Returns
    -------
    df : pd.DataFrame
        The output DataFrame.

    Example
    -------
    Create a transformer to be used in a scikit-learn pipeline.
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> mask = [
    >>>     ("X1", -10, 10),
    >>>     ("X2", -25, 25),
    >>>     ("X3", -5, 5),
    >>>     ("X4", -7, 7),
    >>>     ("X5", -3, 3),
    >>>     ("X6", -2, 2),
    >>>     ("X7", -13, 13),
    >>>     ("X8", -8, 8),
    >>>     ("X9", -9, 9),
    >>>     ("X10", -10, 10),
    >>>     ("X11", -9, 9),
    >>>     ("X12", -16, 17),
    >>>     ("X13", -20, 23)
    >>> ]
    >>> mask = FunctionTransformer(
    >>>     mask_outliers,
    >>>     kw_args={"mask": mask}
    >>> )
    >>> imputer = SimpleImputer()
    >>> imputer_pipeline = make_pipeline(mask, imputer)
    >>> transformer = make_column_transformer(
    >>>     (imputer_pipeline, features),
    >>>     remainder="drop"
    >>> )
    """
    for column, lowvalue, highvalue in mask:
        df[column] = df[column].mask(
            cond=(df[column] <= lowvalue) | (df[column] >= highvalue),
            other=pd.NA
        )
    return pd.DataFrame(data=df)


def main():
    file_predictions = Path("outliers_missing_predictions.csv")
    features = [
        "X1", "X2", "X3", "X4", "X5", "X6", "X7",
        "X8", "X9", "X10", "X11", "X12", "X13", "X14"
    ]
    mask = [
        ("X1", -10, 10),
        ("X2", -25, 25),
        ("X3", -5, 5),
        ("X4", -7, 7),
        ("X5", -3, 3),
        ("X6", -2, 2),
        ("X7", -13, 13),
        ("X8", -8, 8),
        ("X9", -9, 9),
        ("X10", -10, 10),
        ("X11", -9, 9),
        ("X12", -16, 17),
        ("X13", -20, 23)
    ]
    file_new = Path("outliers_missing_new.csv")
    file_data = Path("outliers_missing.csv")
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    target = "Y"
    nrows = 200
    df = pd.read_csv(
        filepath_or_buffer=file_data,
        skip_blank_lines=False,
        nrows=nrows
    )
    df = ds.delete_empty_rows(
        df=df,
        list_columns=[target]
    )
    X = df[features]
    y = df[target]
    df_new = pd.read_csv(
        filepath_or_buffer=file_new,
        skip_blank_lines=False
    )
    X_new = df_new[features]
    print("Workflow 1")
    print()
    mask = FunctionTransformer(
        mask_outliers,
        kw_args={"mask": mask}
    )
    imputer = SimpleImputer()
    imputer_pipeline = make_pipeline(mask, imputer)
    transformer = make_column_transformer(
        (imputer_pipeline, features),
        remainder="drop"
    )
    decision_tree_regressor = DecisionTreeRegressor()
    random_forest_regressor = RandomForestRegressor()
    linear_regression = LinearRegression()
    xgboost_regressor = XGBRegressor()
    lassocv = LassoCV()
    lasso = Lasso()
    selector = SelectFromModel(estimator=decision_tree_regressor)
    hyperparameters1 = {}
    hyperparameters1["transformer"] = [imputer]
    hyperparameters1["transformer__strategy"] = [
        "mean", "median", "most_frequent", "constant"
    ]
    hyperparameters1["selector"] = [
        SelectFromModel(estimator=decision_tree_regressor)
    ]
    hyperparameters1["selector__threshold"] = [None, "mean", "median"]
    hyperparameters1["selector__estimator__criterion"] = [
        "squared_error", "friedman_mse", "absolute_error"
    ]
    hyperparameters1["selector__estimator__splitter"] = ["best", "random"]
    hyperparameters1["selector__estimator__max_features"] = [
        None, "sqrt", "log2"
    ]
    hyperparameters1["regressor"] = [linear_regression]
    hyperparameters2 = {}
    hyperparameters2["transformer"] = [imputer]
    hyperparameters2["transformer__strategy"] = [
        "mean", "median", "most_frequent", "constant"
    ]
    hyperparameters2["selector"] = [SelectFromModel(estimator=lassocv)]
    hyperparameters2["selector__threshold"] = [None, "mean", "median"]
    hyperparameters2["regressor"] = [linear_regression]
    hyperparameters = [hyperparameters1, hyperparameters2]
    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("selector", selector),
            ("regressor", linear_regression)
        ]
    )
    print(pipeline)
    pipeline.fit(
        X=X,
        y=y
    )
    print()
    print("Hyperparamters:")
    print(hyperparameters)
    print()
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=hyperparameters,
        scoring="r2",
        n_jobs=-1,
        cv=10
    )
    grid_search.fit(
        X=X,
        y=y
    )
    print("Best hyperparameters")
    print(grid_search.best_params_)
    print()
    print("Best score")
    print(grid_search.best_score_.round(3))
    print()
    print("Workflow 2")
    print()
    imputer = SimpleImputer(strategy="mean")
    transformer = make_column_transformer(
         (imputer, features),
         remainder="passthrough"
    )
    selector = SelectFromModel(
        estimator=lassocv,
        threshold="median"
    )
    pipeline = make_pipeline(transformer, selector, linear_regression)
    print(pipeline)
    print()
    pipeline.fit(
        X=X,
        y=y
    )
    selected_features = X.columns[selector.get_support()].to_list()
    print("Selected features")
    selected_coefficients = pipeline.named_steps.linearregression.\
        coef_.round(3)
    selected_importances = np.abs(
        pipeline.named_steps.selectfrommodel.estimator_.coef_[
            selector.get_support()
        ]
    ).tolist()
    print(pd.DataFrame(
        list(zip(
            selected_features, selected_importances, selected_coefficients)),
        columns=["Features", "Importance", "Coefficients"]
    ))
    print()
    print("Regression intercept")
    print(pipeline.named_steps.linearregression.intercept_.round(3))
    print()
    print("Cross-validation score")
    print(cross_val_score(
        estimator=pipeline,
        X=X,
        y=y,
        scoring="r2",
        cv=10,
        n_jobs=-1
    ).mean().round(3))
    print()
    predictions_ndarray = grid_search.predict(X=X_new)
    predictions_series = pd.Series(
        data=predictions_ndarray,
        index=X_new.index,
        name="y predictions"
    )
    X_new_predictions = pd.concat(
        objs=[X_new, predictions_series],
        axis="columns"
    )
    X_new_predictions.to_csv(path_or_buf=file_predictions)


if __name__ == "__main__":
    main()
