#! /usr/bin/env python3


'''
Supervised machine learning logistic regression

./titanic_essential_part_1.py
'''

# Import librairies
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline


# Define features and target
features_all = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name']
features_categorical = ['Embarked', 'Sex']
features_text = 'Name'
target = 'Survived'
# Import the training data set into a DataFrame
df = pd.read_csv('http://bit.ly/kaggletrain', nrows=10)
# Create a DataFrame of the features for the training data set
X = df[features_all]
# Create a Series of the target for the training data set
y = df[target]
# Import the testing data set
df_new = pd.read_csv('http://bit.ly/kaggletest', nrows=10)
# Create a DataFrame of the features for the testing data set
X_new = df_new[features_all]
# Create the OneHotEncoder object for encoding categorical columns
ohe = OneHotEncoder()
# Create the CountVectorizer object for encoding text columns
vect = CountVectorizer()
# Create the column transformer object
ct = make_column_transformer(
    (ohe, features_categorical),
    (vect, features_text),
    remainder='passthrough')
# Create the logistic regression object
logreg = LogisticRegression(solver='liblinear', random_state=1)
# Create the pipeline object
pipe = make_pipeline(ct, logreg)
# Call the fit method
pipe.fit(X, y)
# Call the predict method
pipe.predict(X_new)
