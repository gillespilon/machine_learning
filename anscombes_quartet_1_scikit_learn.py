#! /usr/bin/env python3
"""
Example of simple linear regression using scikit-learn

./anscombes_quartet_1.py
"""

from sklearn.linear_model import LinearRegression
import datasense as ds
import pandas as pd

feature = ['x']
target = 'y'
data = pd.read_csv('anscombes_quartet_1.csv')
X = data[feature]
y = data[target]
linreg = LinearRegression()
linreg.fit(X, y)
print('intercept  :', linreg.intercept_)
print('coefficient:', linreg.coef_)
y_predicted = pd.Series(linreg.predict(X))
data_all = X
data_all = pd.concat(
    [data_all, y, y_predicted],
    axis='columns',
    sort=False
).set_axis(
    labels=['x', 'y', 'y_predicted'],
    axis='columns'
)
fig, ax = ds.plot_scatter_line_x_y1_y2(
    X=data_all['x'],
    y1=data_all['y'],
    y2=data_all['y_predicted'],
    figsize=(8, 6)
)
ds.despine(ax)
fig.savefig(
    fname='anscombes_quartet_1.svg',
    format='svg'
)
