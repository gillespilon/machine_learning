#! /usr/bin/env python3
"""
Example of scikit-learn for supervised machine learning, Xs are measurements

Create training and testing data sets
Create a workflow pipeline
Add a column transformer object
Add a feature selection object
Add a regression object
Fit a model
Set hyperparameters for tuning
Tune the model using grid search cross validation

time -f '%e' ./lunch_and_learn_essential.py | tee lunch_and_learn_essential.txt
time -f '%e' ./lunch_and_learn_essential.py > lunch_and_learn_essential.txt
./lunch_and_learn_essential.py > lunch_and_learn_essential.txt
./lunch_and_learn_essential.py
"""

from multiprocessing import Pool
from typing import List, Tuple
from datetime import datetime
import math

from sklearn.model_selection import cross_validate, cross_val_score,\
        cross_val_predict, GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, SCORERS
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from sklearn import set_config
import datasense as ds
import pandas as pd
import numpy as np

colour1 = '#0077bb'
colour2 = '#33bbee'
pd.options.display.max_rows = None
pd.options.display.max_columns = None
file_name = 'lunch_and_learn.csv'
graph_name = 'predicted_versus_measured'
nrows = 5000
target = 'Y'
features = [
    'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7',
    'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14'
]
percent_empty_features = 60.0
set_config(display='diagram')
label_predicted = 'Predicted'
label_measured = 'Measured'
title = 'Predicted versus Measured'
figure_width_height = (8, 6)


def feature_percent_empty(
    df: pd.DataFrame,
    cols: List[str],
    limit: float
) -> List[str]:
    '''
    Remove features that have NaN > limit
    '''
    num_rows = df.shape[0]
    return [col for col in cols if
            ((df[col].isna().sum() / num_rows * 100) <= limit)]


def despine(ax: axes.Axes) -> None:
    """
    Remove the top and right spines of a graph.

    Parameters
    ----------
    ax : axes.Axes

    Example
    -------
    >>> despine(ax)
    """
    for spine in 'right', 'top':
        ax.spines[spine].set_visible(False)


def plot_time_series(
    yvals: pd.Series,
    ytext: str,
    figwh: Tuple[int, int],
    graphname: str
) -> None:
    '''
    Scatter plot of y versus sample order
    '''
    fig = plt.figure(figsize=figwh)
    ax = fig.add_subplot(111)
    ax.plot(yvals, marker='.', linestyle='', color=colour1)
    ax.set_ylabel(ytext)
    ax.set_ylabel(ytext)
    ax.set_title('Time Series')
    despine(ax)
    plt.savefig(f'{graphname}_time_series_{ytext}.svg')
    # If you wish to see the graphs inline,
    # comment the next line
    # Otherwise graph files are saved to the current working directory
    plt.close()


def plot_scatter_line(
    yvals: pd.Series,
    xvals: np.ndarray,
    ytext: str,
    xtext: str,
    titletext: str,
    figwh: Tuple[int, int],
    graphname: str
) -> None:
    '''
    Scatter plot of y versus x
    Line of perfect fit
    '''
    fig = plt.figure(figsize=figwh)
    ax = fig.add_subplot(111)
    ax.plot(yvals, xvals, marker='.', linestyle='', color=colour1)
    ax.plot([yvals.min(), yvals.max()], [yvals.min(), yvals.max()],
            marker=None, linestyle='-', color=colour2)
    ax.set_ylabel(ytext)
    ax.set_xlabel(xtext)
    ax.set_title(titletext)
    despine(ax)
    plt.savefig(f'{graphname}_scatter.svg')


def plot_line_line(
    yvals1: pd.Series,
    yvals2: np.ndarray,
    yvals1text: str,
    yvals2text: str,
    titletext: str,
    figwh: Tuple[int, int],
    graphname: str
) -> None:
    '''
    Two line plots of y1, y2
    '''
    fig = plt.figure(figsize=figwh)
    ax = fig.add_subplot(111)
    ax.plot(
        yvals1, marker='.', linestyle='-',
        color=colour1, label=yvals1text
    )
    ax.plot(
        yvals2, marker='.', linestyle='-',
        color=colour2, label=yvals2text
    )
    ax.set_title(titletext)
    ax.legend(frameon=False)
    despine(ax)
    plt.savefig(f'{graphname}_lines.svg')


# Cleaning the data
# Data should be cleaned before fitting a model. A simple example of graphing
# each feature in sample order and replacing outliers with NaN is shown.
# Read the data file into a pandas DataFrame
data = ds.read_file(
    file_name=file_name,
    # nrows=nrows
)


# Plot target versus features
for feature in features:
    plot_time_series(
        data[feature], feature, figure_width_height, graph_name
    )


# Set lower and upper values to remove outliers
mask_values = [
    ('X1', -20, 20),
    ('X2', -25, 25),
    ('X3', -5, 5),
    ('X4', -10, 10),
    ('X5', -3, 3),
    ('X6', -5, 5),
    ('X7', -13, 13),
    ('X8', -9, 15),
    ('X9', -17, 15),
    ('X10', -16, 15),
    ('X11', -16, 17),
    ('X12', -16, 17),
    ('X13', -20, 23)
]


# Replace outliers with NaN
for column, lowvalue, highvalue in mask_values:
    data[column] = data[column].mask(
        (data[column] <= lowvalue) |
        (data[column] >= highvalue)
    )


# Delete rows if target is NaN
data = data.dropna(subset=[target])


# Remove features if > percent_empty_features
# Do not impute for missing values if too many missing
features = feature_percent_empty(data, features, percent_empty_features)
print('Features')
print(features)


# Create training and testing data sets
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
print('Workflow 1')
# Impute using the mean
# Select features using SelectFromModel(DecisionTreeRegressor)
# Fit with LinearRegression


# Create the imputer object with
# the default hyperparameter settings
imp = SimpleImputer()


# Create the column transformer object
ct = make_column_transformer(
    (imp, features),
    remainder='passthrough'
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
        ('transformer', ct),
        ('selector', selection),
        ('regressor', linreg)
    ]
)
print()
print(pipe)


# Determine the linear regression model
pipe.fit(X_train, y_train)


# Show the selected features
print()
print('Selected features')
print(X_all.columns[selection.get_support()])


# Display the regression intercept
print()
print('Regression intercept')
print(pipe.named_steps.regressor.intercept_.round(3))


# Display the regression coefficients of the features
print()
print('Regression coefficients')
print(pipe.named_steps.regressor.coef_.round(3))


# Cross-validate the updated pipeline
print()
print('Cross-validation score')
print(cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1).mean().round(3))


# Set the hyperparameters for optimization
# Create a dictionary
# The dictionary key is the step name, followed by two underscores,
# followed by the hyperparameter name
# The dictionary value is the list of values to try per hyperparameter
# 4 x 3 x 3 x 2 = 72
hyperparams = [
    {
        'transformer': [imp],
        'transformer__strategy': [
            'mean', 'median', 'most_frequent', 'constant'
        ],
        'selector': [SelectFromModel(estimator=dtr_selection)],
        'selector__threshold': [None, 'mean', 'median'],
        'selector__estimator__criterion': [
            'mse', 'friedman_mse', 'mae'
        ],
        # 'selector__estimator__splitter': ['best', 'random'],
        # 'selector__estimator__max_features': [
        #     None, 'auto', 'sqrt', 'log2'
        # ],
        # 'selector__estimator__max_leaf_nodes': [None, 2, 4, 6],
        'regressor': [linreg],
        'regressor__normalize': [False, True]
    },

]
'''hyperparams.append(
    {
        'transformer': [imp],
        'transformer__strategy': [
            'mean', 'median', 'most_frequent', 'constant'
        ],
        'selector': [SelectFromModel(estimator=linreg_selection)],
        'selector__threshold': [None, 'mean', 'median'],
        'selector__estimator__normalize': [False, True],
        'regressor': [linreg],
        'regressor__normalize': [False, True]
    },
)'''
'''
hyperparams.append(
   {
        'transformer': [imp],
        'transformer__strategy': [
            'mean', 'median', 'most_frequent', 'constant'
        ],
        'selector': [SelectFromModel(estimator=lasso_selection)],
        'selector__threshold': [None, 'mean', 'median'],
        'selector__estimator__normalize': [False, True],
        'regressor': [linreg],
        'regressor__normalize': [False, True]
    },
)
'''
# 4 x 3 x 2 x 2 = 48
hyperparams.append(
   {
        'transformer': [imp],
        'transformer__strategy': [
            'mean', 'median', 'most_frequent', 'constant'
        ],
        'selector': [SelectFromModel(estimator=lassocv_selection)],
        'selector__threshold': [None, 'mean', 'median'],
        'selector__estimator__normalize': [False, True],
        'regressor': [linreg],
        'regressor__normalize': [False, True]
    },
)
'''
hyperparams.append(

    {

        'transformer': [imp],

        'transformer__strategy': [

            'mean', 'median', 'most_frequent', 'constant'

        ],

        'selector': [SelectFromModel(estimator=rfr_selection)],

        'selector__threshold': [None, 'mean', 'median'],

        'selector__estimator__criterion': ['mse', 'mae'],

        'regressor': [linreg],

        'regressor__normalize': [False, True]

    },

)
'''


# Perform a grid search
grid = GridSearchCV(pipe, hyperparams, n_jobs=-1, cv=5)
grid.fit(X_train, y_train)


# Present the results
pd.DataFrame(grid.cv_results_).sort_values('rank_test_score')


# Access the best score
print()
print('Hyperparameter optimization')
print('Best score')
print(grid.best_score_.round(3))


# Access the best hyperparameters
print()
print('Best hyperparameters')
print(grid.best_params_)


# Workflow 2
print()
print('Workflow 2')
# Impute using the mean
# Select features using SelectFromModel(LassoCV())
# Fit with LinearRegression()
# Create the imputer object
imp = SimpleImputer()
# Create the column transformer object
ct = make_column_transformer(
     (imp, features),
     remainder='passthrough'
)
# Create the object to use for feature selection
lassocv_selection = LassoCV()
# Create the feature selection object
selection = SelectFromModel(
    estimator=lassocv_selection,
    threshold='median'
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
print('Selected features')
print(selected_features)
selected_coefficients = pipe.named_steps.linearregression.coef_.round(3)
selected_importances = np.abs(
    pipe.named_steps.selectfrommodel.estimator_.coef_[selection.get_support()]
).tolist()
print()
print(pd.DataFrame(
    list(zip(
        selected_features, selected_importances, selected_coefficients)),
    columns=['Features', 'Importance', 'Coefficients']
))
# Display the regression intercept
print()
print('Regression intercept')
print(pipe.named_steps.linearregression.intercept_.round(3))
print()
print('Cross-validation score')
print(cross_val_score(
    pipe, X_train, y_train, cv=5, n_jobs=-1,
    scoring='r2'
).mean().round(3))
# Calculate predicted values
predicted = cross_val_predict(pipe, X_all, y_all, cv=5, n_jobs=-1)
mse = mean_squared_error(y_all, predicted)
print()
print('Mean squared error')
print(mse.round(3))
print()
print('Root mean squared error')
print(round(math.sqrt(mse), 3))
# Plot predicted versus measured
plot_scatter_line(
    y_all, predicted, label_predicted, label_measured, title,
    figure_width_height, graph_name
)
# Plot predicted versus measured
plot_line_line(y_all, predicted, label_measured, label_predicted, title,
               figure_width_height, graph_name)
