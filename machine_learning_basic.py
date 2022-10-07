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
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn import set_config
import datasense as ds
import pandas as pd
import numpy as np

features = [
    "X1", "X2", "X3", "X4", "X5", "X6", "X7",
    "X8", "X9", "X10", "X11", "X12", "X13", "X14"
]
output_url = "machine_learning_basic.html"
header_title = "machine_learning_basic"
graph_name = "predicted_versus_measured"
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


def plot_scatter_y(t: pd.Series) -> NoReturn:
    y, feature = t
    fig, ax = ds.plot_scatter_y(y=y, figsize=figsize)
    ax.set_ylabel(ylabel=feature)
    ax.set_title(label="Time Series")
    ds.despine(ax=ax)
    fig.savefig(fname=f"time_series_{feature}.svg", format="svg")
    ds.html_figure(
        file_name=f"time_series_{feature}.svg",
        caption=f"time_series_{feature}.svg"
    )


start_time = time.time()
original_stdout = ds.html_begin(
    output_url=output_url,
    header_title=header_title,
    header_id=header_id
)
ds.page_break()
print("<pre style='white-space: pre-wrap;'>")
# Cleaning the data
# Data should be cleaned before fitting a model. A simple example of graphing
# each feature in sample order and replacing outliers with NaN is shown.
# Read the data file into a pandas DataFrame
data = ds.read_file(
    file_name=file_name,
    nrows=nrows
)
# Plot target versus features
# With multiprocessing
t = ((data[feature], feature) for feature in features)
with Pool() as pool:
    for _ in pool.imap_unordered(plot_scatter_y, t):
        pass
# Without multiprocessing
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

# Replace outliers with NaN
for column, lowvalue, highvalue in mask_values:
    data[column] = data[column].mask(
        (data[column] <= lowvalue) |
        (data[column] >= highvalue)
    )

# Delete rows if target is NaN
data = data.dropna(subset=[target])

ds.page_break()
print("Features before threshold")
print(features)
print()
# Remove features if number empty cells > threshold
features = ds.feature_percent_empty(
    df=data,
    columns=features,
    threshold=percent_empty_features
)
print("Features after threshold")
print(features)

# Create training and testing datasets
X_all = data[features]
y_all = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.33, random_state=42
)

# Machine learning workflow
# A typical workflow involves several, sequential steps:
# - Column transformation such as imputing missing values
# - Feature selection
# - Modeling
# These steps are embedded in a workflow method called a pipeline, which is
# simply a series of sequential steps. The output of each step is passed to
# the next step.

# Workflow 1
print()
print("Workflow 1")
# Impute using the mean
# Select features using SelectFromModel(DecisionTreeRegressor)
# Fit with LinearRegression

# Create the imputer object with
# the default hyperparameter settings
imp = SimpleImputer()

# Create the column transformer object
ct = make_column_transformer(
    (imp, features),
    remainder="passthrough"
)

# Create objects to use for feature selection with
# the default hyperparameter settings
linreg_selection = LinearRegression()
dtr_selection = DecisionTreeRegressor()
lasso_selection = Lasso()
lassocv_selection = LassoCV()
rfr_selection = RandomForestRegressor()

# Create the feature selection object
selection = SelectFromModel(estimator=dtr_selection)

# Create an object to use for regression with
# the default hyperparameter settings
linreg = LinearRegression()
dtr = DecisionTreeRegressor()
lasso = Lasso()
lassocv = LassoCV()
rfr = RandomForestRegressor()
xgb = XGBRegressor()

# Create the workflow object
pipe = Pipeline(
    steps=[
        ("transformer", ct),
        ("selector", selection),
        ("regressor", linreg)
    ]
)
print()
print(pipe)

# Determine the linear regression model
pipe.fit(X_train, y_train)

# Show the selected features
print()
print("Selected features")
print(X_all.columns[selection.get_support()])

# Display the regression intercept
print()
print("Regression intercept")
print(pipe.named_steps.regressor.intercept_.round(3))

# Display the regression coefficients of the features
print()
print("Regression coefficients")
print(pipe.named_steps.regressor.coef_.round(3))

# Cross-validate the updated pipeline
print()
print("Cross-validation score")
print(cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1).mean().round(3))

# Set the hyperparameters for optimization
# Create a dictionary
# The dictionary key is the step name, followed by two underscores,
# followed by the hyperparameter name
# The dictionary value is the list of values to try per hyperparameter
# 4 x 3 x 3 x 2 = 72
hyperparams = [
    {
        "transformer": [imp],
        "transformer__strategy": [
            "mean", "median", "most_frequent", "constant"
        ],
        "selector": [SelectFromModel(estimator=dtr_selection)],
        "selector__threshold": [None, "mean", "median"],
        "selector__estimator__criterion": [
            "squared_error", "friedman_mse", "absolute_error"
        ],
        # "selector__estimator__splitter": ["best", "random"],
        # "selector__estimator__max_features": [
        #     None, "auto", "sqrt", "log2"
        # ],
        # "selector__estimator__max_leaf_nodes": [None, 2, 4, 6],
        "regressor": [linreg],
        # "regressor__normalize": [False, True]
    },

]
"""hyperparams.append(
    {
        "transformer": [imp],
        "transformer__strategy": [
            "mean", "median", "most_frequent", "constant"
        ],
        "selector": [SelectFromModel(estimator=linreg_selection)],
        "selector__threshold": [None, "mean", "median"],
        "selector__estimator__normalize": [False, True],
        "regressor": [linreg],
        "regressor__normalize": [False, True]
    },
)"""
"""
hyperparams.append(
   {
        "transformer": [imp],
        "transformer__strategy": [
            "mean", "median", "most_frequent", "constant"
        ],
        "selector": [SelectFromModel(estimator=lasso_selection)],
        "selector__threshold": [None, "mean", "median"],
        "selector__estimator__normalize": [False, True],
        "regressor": [linreg],
        "regressor__normalize": [False, True]
    },
)
"""
# 4 x 3 x 2 x 2 = 48
hyperparams.append(
   {
        "transformer": [imp],
        "transformer__strategy": [
            "mean", "median", "most_frequent", "constant"
        ],
        "selector": [SelectFromModel(estimator=lassocv_selection)],
        "selector__threshold": [None, "mean", "median"],
        # "selector__estimator__normalize": [False, True],
        "regressor": [linreg],
        # "regressor__normalize": [False, True]
    },
)
"""
hyperparams.append(

    {

        "transformer": [imp],

        "transformer__strategy": [

            "mean", "median", "most_frequent", "constant"

        ],

        "selector": [SelectFromModel(estimator=rfr_selection)],

        "selector__threshold": [None, "mean", "median"],

        "selector__estimator__criterion": ["squared_error", "absolute_error"],

        "regressor": [linreg],

        "regressor__normalize": [False, True]

    },

)
"""

# Perform a grid search
grid = GridSearchCV(pipe, hyperparams, n_jobs=-1, cv=5)
grid.fit(X_train, y_train)

# Present the results
pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")

# Access the best score
print()
print("Hyperparameter optimization")
print("Best score")
print(grid.best_score_.round(3))

# Access the best hyperparameters
print()
print("Best hyperparameters")
print(grid.best_params_)

# Workflow 2
ds.page_break()
print("Workflow 2")
# Impute using the mean
# Select features using SelectFromModel(LassoCV())
# Fit with LinearRegression()
# Create the imputer object
imp = SimpleImputer()
# Create the column transformer object
ct = make_column_transformer(
     (imp, features),
     remainder="passthrough"
)
# Create the object to use for feature selection
lassocv_selection = LassoCV()
# Create the feature selection object
selection = SelectFromModel(
    estimator=lassocv_selection,
    threshold="median"
)
# Create objects to use for regression
linreg = LinearRegression()
# Create the workflow object
pipe = make_pipeline(ct, selection, linreg)
print()
print(pipe)
# Determine the linear regression model
pipe.fit(X_train, y_train)
# Show the selected features =
# selected = pd.DataFrame(X.columns[selection.get_support()])
selected_features = X_all.columns[selection.get_support()].to_list()
print()
print("Selected features")
print(selected_features)
selected_coefficients = pipe.named_steps.linearregression.coef_.round(3)
selected_importances = np.abs(
    pipe.named_steps.selectfrommodel.estimator_.coef_[selection.get_support()]
).tolist()
print()
print(pd.DataFrame(
    list(zip(
        selected_features, selected_importances, selected_coefficients)),
    columns=["Features", "Importance", "Coefficients"]
))
# Display the regression intercept
print()
print("Regression intercept")
print(pipe.named_steps.linearregression.intercept_.round(3))
print()
print("Cross-validation score")
print(cross_val_score(
    pipe, X_train, y_train, cv=5, n_jobs=-1,
    scoring="r2"
).mean().round(3))
# Calculate predicted values
predicted = cross_val_predict(pipe, X_all, y_all, cv=5, n_jobs=-1)
mse = mean_squared_error(y_all, predicted)
print()
print("Mean squared error")
print(mse.round(3))
print()
print("Root mean squared error")
print(round(math.sqrt(mse), 3))
ds.page_break()
# Scatter plot of predicted versus measured
fig, ax = ds.plot_scatter_x_y(
    X=y_all,
    y=predicted,
    figsize=figsize
)
ax.plot(
    [y_all.min(), y_all.max()],
    [y_all.min(), y_all.max()],
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
# Line plot of predicted versus measured
fig, ax = ds.plot_line_line_y1_y2(
    y1=y_all,
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
