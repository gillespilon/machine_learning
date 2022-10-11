#! /usr/bin/env python3
"""
Example of scikit-learn for supervised machine learning, Xs are measurements

Create training and testing datasets
Create a workflow pipeline
Add a column transformer object
Add a feature selection object
Add a regression object
Fit a model
Set hyperparameters for tuning
Tune the model using grid search cross validation

time -f "%e" ./machine_learning_basic.py | tee machine_learning_basic.txt
time -f "%e" ./machine_learning_basic.py > machine_learning_basic.txt
./machine_learning_basic.py > machine_learning_basic.txt
./machine_learning_basic.py
"""

from multiprocessing import Pool
from typing import NoReturn
import time
import math

from sklearn.model_selection import cross_val_score, cross_val_predict,\
    GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn import set_config
import datasense as ds
import pandas as pd
import numpy as np


def plot_scatter_y(t: pd.Series) -> NoReturn:
    y, feature = t
    fig, ax = ds.plot_scatter_y(
        y=y,
        figsize=figsize
    )
    ax.set_ylabel(ylabel=feature)
    ax.set_title(label="Time Series")
    ds.despine(ax=ax)
    fig.savefig(
        fname=f"time_series_{feature}.svg",
        format="svg"
    )
    ds.html_figure(
        file_name=f"time_series_{feature}.svg",
        caption=f"time_series_{feature}.svg"
    )


def main():
    global figsize
    features = [
        "X1", "X2", "X3", "X4", "X5", "X6", "X7",
        "X8", "X9", "X10", "X11", "X12", "X13", "X14"
    ]
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
    output_url = "machine_learning_basic.html"
    graph_name = "predicted_versus_measured"
    header_title = "machine_learning_basic"
    colour1, colour2 = "#0077bb", "#33bbee"
    pd.options.display.max_columns = None
    header_id = "lunch-and-learn-basic"
    title = "Predicted versus Measured"
    pd.options.display.max_rows = None
    file_name = "machine_learning.csv"
    percent_empty_features = 60.0
    set_config(display="diagram")
    label_predicted = "Predicted"
    label_measured = "Measured"
    figsize = (8, 4.5)
    target = "Y"
    nrows = 200
    start_time = time.time()
    original_stdout = ds.html_begin(
        output_url=output_url,
        header_title=header_title,
        header_id=header_id
    )
    ds.page_break()
    print("<pre style='white-space: pre-wrap;'>")
    data = ds.read_file(
        file_name=file_name,
        nrows=nrows
    )
    # data = pd.read_csv(
    #     filepath_or_buffer=file_data,
    #     nrows=nrows
    # )
    # Plot target versus features with multiprocessing
    t = ((data[feature], feature) for feature in features)
    with Pool() as pool:
        for _ in pool.imap_unordered(plot_scatter_y, t):
            pass
    # Plot target versus features without multiprocessing
    # for feature in features:
    #     fig, ax = ds.plot_scatter_y(
    #         y=data[feature],
    #         figsize=figsize
    #     )
    #     ax.set_ylabel(ylabel=feature)
    #     ax.set_title(label="Time Series")
    #     ds.despine(ax=ax)
    #     fig.savefig(
    #         fname=f"time_series_{feature}.svg",
    #         format="svg"
    #     )
    #     ds.html_figure(
    #         file_name=f"time_series_{feature}.svg",
    #         caption=f"time_series_{feature}.svg"
    #     )
    # Replace outliers with NaN outside of scikit-learn
    # TODO: The next five lines should be done within scikit-learn.
    for column, lowvalue, highvalue in mask_values:
        data[column] = data[column].mask(
            (data[column] <= lowvalue) |
            (data[column] >= highvalue)
        )
    data = data.dropna(subset=[target])
    ds.page_break()
    print("Features before threshold")
    print(features)
    print()
    # Remove features if number empty cells > threshold
    # features = ds.feature_percent_empty(
    #     df=data,
    #     columns=features,
    #     threshold=percent_empty_features
    # )
    # print("Features after threshold")
    # print(features)
    # Create training and testing datasets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42
    )
    print("Workflow 1")
    print()
    imputer = SimpleImputer()
    transformer = make_column_transformer(
        (imputer, features),
        remainder="passthrough"
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
    # hyperparameters3 = {}
    # hyperparameters3["transformer"] = [imputer]
    # hyperparameters3["transformer__strategy"] = [
    #     "mean", "median", "most_frequent", "constant"
    # ]
    # hyperparameters3["selector"] = [
    #     SelectFromModel(estimator=linear_regression)
    # ]
    # hyperparameters3["selector__threshold"] = [None, "mean", "median"]
    # hyperparameters3["regressor"] = [linear_regression]
    # hyperparameters4 = {}
    # hyperparameters4["transformer"] = [imputer]
    # hyperparameters4["transformer__strategy"] = [
    #     "mean", "median", "most_frequent", "constant"
    # ]
    # hyperparameters4["selector"] = [SelectFromModel(estimator=lasso)]
    # hyperparameters4["selector__threshold"] = [None, "mean", "median"]
    # hyperparameters4["regressor"] = [linear_regression]
    # hyperparameters5 = {}
    # hyperparameters5["transformer"] = [imputer]
    # hyperparameters5["transformer__strategy"] = [
    #     "mean", "median", "most_frequent", "constant"
    # ]
    # hyperparameters5["selector"] = [
    #     SelectFromModel(estimator=random_forest_regressor)
    # ]
    # hyperparameters5["selector__threshold"] = [None, "mean", "median"]
    # hyperparameters5["selector__estimator__criterion"] = [
    #     "squared_error", "absolute_error"
    # ]
    # hyperparameters5["regressor"] = [linear_regression]
    hyperparameters = [hyperparameters1, hyperparameters2]
    # Create the workflow object
    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("selector", selector),
            ("regressor", linear_regression)
        ]
    )
    print(pipeline)
    pipeline.fit(
        X=X_train,
        y=y_train
    )
    print()
    print("hyperparamters:")
    print(hyperparameters)
    print()
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=hyperparameters,
        n_jobs=-1,
        cv=5
    )
    grid.fit(
        X=X_train,
        y=y_train
    )
    print("Best hyperparameters")
    print(grid.best_params_)
    print()
    print("Best score")
    print(grid.best_score_.round(3))
    print()
    print("Selected features")
    print(X.columns[selector.get_support()])
    print()
    print("Regression intercept")
    print(pipeline.named_steps.regressor.intercept_.round(3))
    print()
    print("Regression coefficients")
    print(pipeline.named_steps.regressor.coef_.round(3))
    ds.page_break()
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
        X=X_train,
        y=y_train
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
        X=X_train,
        y=y_train,
        cv=5,
        n_jobs=-1,
        scoring="r2"
    ).mean().round(3))
    print()
    predicted = cross_val_predict(
        estimator=pipeline,
        X=X,
        y=y,
        cv=5,
        n_jobs=-1
        )
    mse = mean_squared_error(y, predicted)
    print("Mean squared error")
    print(mse.round(3))
    print()
    print("Root mean squared error")
    print(round(math.sqrt(mse), 3))
    print()
    ds.page_break()
    fig, ax = ds.plot_scatter_x_y(
        X=y,
        y=predicted,
        figsize=figsize
    )
    ax.plot(
        [y.min(), y.max()],
        [y.min(), y.max()],
        marker=None,
        linestyle="-",
        color=colour2
    )
    ax.set_ylabel(ylabel=label_predicted)
    ax.set_xlabel(xlabel=label_measured)
    ax.set_title(label=title)
    ds.despine(ax=ax)
    fig.savefig(
        fname=f"{graph_name}_scatter.svg",
        format="svg"
    )
    ds.html_figure(
        file_name=f"{graph_name}_scatter.svg",
        caption=f"{graph_name}_scatter.svg"
    )
    fig, ax = ds.plot_line_line_y1_y2(
        y1=y,
        y2=predicted,
        figsize=figsize,
        labellegendy1=label_measured,
        labellegendy2=label_predicted
    )
    ax.legend(frameon=False)
    ax.set_title(label=title)
    ds.despine(ax=ax)
    fig.savefig(
        fname=f"{graph_name}_lines.svg",
        format="svg"
    )
    ds.html_figure(
        file_name=f"{graph_name}_lines.svg",
        caption=f"{graph_name}_lines.svg"
    )
    stop_time = time.time()
    ds.page_break()
    ds.report_summary(
        start_time=start_time,
        stop_time=stop_time
    )
    print("</pre>")
    ds.html_end(
        original_stdout=original_stdout,
        output_url=output_url
    )


if __name__ == "__main__":
    main()
