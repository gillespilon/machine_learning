#! /usr/bin/env python3


'''
Example of simple linear regression

./anscombes_quartet_1_tensorflow.py
'''


# Import librairies
import pandas as pd
import tensorflow as tf
import datasense as ds


# Define feature and target
feature = 'x'
target = 'y'
# Import data into a DataFrame
data = pd.read_csv('anscombes_quartet_1.csv')
xs = data[[feature]].to_numpy()
ys = data[[target]].to_numpy()
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=100)
print(model.predict([10.0]))
