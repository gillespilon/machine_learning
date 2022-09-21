#! /usr/bin/env python3
"""
Master machine learning with scikit-learn
"""

from pathlib import Path
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
import datasense as ds
import pandas as pd
import sklearn


def main():
    print("Master machine learning with scikit-learn")
    print()
    print("installed scikit-learn version:", sklearn.__version__)
    print("installed pandas version:      ", pd.__version__)
    print()
    print("2.1 Loading and exploring a dataset")
    print()
    df = pd.read_csv(filepath_or_buffer="titanic_train.csv", nrows=10)
    print("Explore titanic_train.csv")
    print()
    print("Create df:")
    print()
    print(df)
    # Use intuition to select two features
    # Create the X DataFrame
    X = df[["Parch", "Fare"]]
    print()
    print("Create X DataFrame:")
    print(X)
    print()
    # Create the y Series
    y = df["Survived"]
    print("Create y Series for multiclass target:")
    print(y)
    print()
    # Check the shapes of X, y
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()
    print("2.2 Building and evaluate a model")
    print()
    print("Create model, evaluation metric for multiclass classification")
    print()
    logreg = LogisticRegression(solver="liblinear", random_state=1)
    crossvalscore = cross_val_score(
        estimator=logreg, X=X, y=y, cv=3, scoring="accuracy"
    ).mean()
    print("Cross-validation score:", crossvalscore)
    print()
    print("2.3 Using the model to make predictions")
    print()
    logreg.fit(X=X, y=y)
    # df_new is not strictly necessary, can use df, X
    df_new = pd.read_csv(filepath_or_buffer="titanic_train.csv", nrows=10)
    print("Create df_new:")
    print()
    print(df_new)
    print()
    X_new = df_new[["Parch", "Fare"]]
    print("Create X_new:")
    print()
    print(X_new)
    print()
    logreg_predict = logreg.predict(X=X_new)
    # logreg_predict = logreg.predict(X)
    print("logreg.predict:")
    print(logreg_predict)
    print()
    print("2.7 Add model's predictions to the X_new DataFrame")
    print()
    predictions = pd.Series(
        data=logreg_predict, index=X_new.index, name='Prediction'
    )
    X_new_predictions = pd.concat(objs=[X_new, predictions], axis='columns')
    print("Create X_new with predictions:")
    print()
    print(X_new_predictions)
    print()
    print("2.8 Determine the confidence level of each prediction")
    print()
    print("Probabilities of 0, 1:")
    print()
    print(logreg.predict_proba(X=X_new))
    print()
    print("Probabilities of 1:")
    print()
    print(logreg.predict_proba(X=X_new)[:, 1])
    print()
    print("2.10 Show all of the model parameters")
    print()
    print("logreg parameters:")
    print()
    print(logreg.get_params())
    print()
    # 2.12 If you need to shuffle the samples when using cross-validation
    # Used when samples are ordered and shuffling is needed
    # from sklearn.model_selection import StratifiedKFold
    # kf = StratifiedKFold(3, shuffle=True, random_state=1)
    # cross_val_score(estimate=logreg, X=X, y=y, cv=kf, scoring = "accuracy")
    print("3.1 Introduction to one-hot encoding")
    print("3.3 One-hot encoding of multiple features")
    print()
    ohe = OneHotEncoder(sparse=False)
    ohe.fit_transform(X=df[["Embarked", "Sex"]])
    print("Embarked, Sex categories:")
    print()
    print(ohe.categories_)
    print()


if __name__ == "__main__":
    main()
