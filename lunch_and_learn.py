#! /usr/bin/env python3
"""
Python scikit-learn Machine Learning Workflow

In this example I use a simulated data set that has seven features (X1-X7)
that affect the target (Y) and six features (X8-X13) that have zero effect
on the target.

pandas
- Use mask to replace outliers with NaN

scikit-learn
- Impute missing values (NaN) with averages
- Create training and testing data sets
- Perform linear regression for a dependent variable of type float (target)
  and independent variables of type float or integer (features)
- Create a workflow pipeline
- Perform n-fold cross validation on the entire pipeline
- Perform feature selection
- Tune the hyperparameters of the model using grid search

time -f '%e' ./lunch_and_learn.py | tee lunch_and_learn.txt
time -f '%e' ./lunch_and_learn.py > lunch_and_learn.txt
./lunch_and_learn.py > lunch_and_learn.txt
./lunch_and_learn.py
"""

from datetime import datetime

from sklearn.feature_selection import SelectFromModel
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import\
        cross_val_score,\
        GridSearchCV,\
        train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import datasense as ds
import pandas as pd
import numpy as np


pd.options.display.max_rows = None
pd.options.display.max_columns = None
filename = 'lunch_and_learn.csv'
target = 'Y'
features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7',
            'X8', 'X9', 'X10', 'X11', 'X12', 'X13']
data = pd.read_csv(filename)
print('Determine the number of rows and columns\n')
print(data.shape)
print('\nCheck for missing values in the data set\n')
print(data.isna().sum())
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
for column, lowvalue, highvalue in mask_values:
    data[column] = data[column].mask(
        (data[column] <= lowvalue) |
        (data[column] >= highvalue)
    )
# Describe the feature columns
# for column in features:
#     print(column)
#     result = ds.nonparametric_summary(data[column])
#     print(result, '\n')
# Create training and testing data sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
print('\nCheck for missing values in the data set\n')
print(data.isna().sum())
# Create the imputer object
imp = SimpleImputer()
# Create the column transformer object
ct = make_column_transformer(
     (imp, features),
     remainder='passthrough'
)
# Create a regression object to use for the feature selection object
linreg_selection = LinearRegression(fit_intercept=True)
# Create a feature selection object
selection = SelectFromModel(linreg_selection, threshold='median')
# Create the linear regression object
linreg = LinearRegression(fit_intercept=True)
# Create the workflow object
pipe = make_pipeline(ct, selection, linreg)
# Determine the linear regression model for the training data set
pipe.fit(X_train, y_train)
# Set the hyperparameters for optimization
hyperparams = {}
hyperparams['columntransformer__simpleimputer__strategy'] = \
    ['mean', 'median', 'most_frequent', 'constant']
hyperparams['selectfrommodel__threshold'] = [None, 'mean', 'median']
hyperparams['linearregression__normalize'] = [False, True]
# Perform a grid search
grid = GridSearchCV(pipe, hyperparams, cv=5)
grid.fit(X_train, y_train)
# Present the results
print('\nGrid search results')
print(pd.DataFrame(grid.cv_results_).sort_values('rank_test_score'))
# Access the best score
print('\nBest score')
print(grid.best_score_.round(3))
# Access the best hyperparameters
print('\nBest hyperparameters')
print(grid.best_params_)
# Display the regression coefficients of the features
print('\nRegression coefficients')
print(pipe.named_steps.linearregression.coef_.round(3))
# Display the regression intercept
print('\nIntercept')
print(pipe.named_steps.linearregression.intercept_.round(3))
# Show the selected features
print('\nSelected features')
print(X.columns[selection.get_support()])
# Cross-validate the updated pipeline
print('\nCross-validate score')
print(cross_val_score(pipe, X_train, y_train, cv=5).mean().round(3))
