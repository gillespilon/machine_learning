#! /usr/bin/env python3


'''
Example of simple linear regression

./anscombes_quartet_1.py
'''

# Import librairies
import pandas as pd
from sklearn.linear_model import LinearRegression
import datasense as ds


# Define feature and target
feature = ['x']
target = 'y'
# Import data into a DataFrame
data = pd.read_csv('anscombes_quartet_1.csv')
# Create a DataFrame of the feature
X = data[feature]
# Create a Series of the target
y = data[target]
# Create the linear regression object
linreg = LinearRegression()
# Call the fit method
linreg.fit(X, y)
# Display the regression intercept
print(linreg.intercept_)
# Display the regression coefficient
print(linreg.coef_)
# Call the predict method to create an array of predictions
y_predicted = linreg.predict(X)
# Convert the array to a Series
y_predicted = pd.Series(y_predicted)
# Combine all into a DataFrame
data_all = X
data_all = pd.concat([data_all, y, y_predicted], axis='columns', sort=False)
data_all.columns = ['x', 'y', 'y_predicted']
graph = ds.plot_scatter_line_x_y1_y2(data_all, 'x', 'y', 'y_predicted', (8, 6))
graph.figure.savefig('anscombes_quartet_1.svg', format='svg')
