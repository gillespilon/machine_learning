#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Building an effective machine learning
workflow with scikit-learn"
"""

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import datasense as ds
import pandas as pd
# import sklearn


def main():
    # print(sklearn.__version__)
    features = ["Parch", "Fare", "Embarked", "Sex"]
    one_hot_encoder_features = ["Embarked", "Sex"]
    passthrough_features = ["Parch", "Fare"]
    features_two = ["Parch", "Fare"]
    # kaggle training dataset
    file_train = Path("titanic_train.csv")
    # kaggle test dataset`
    file_new = Path("titanic_new.csv")
    vectorizer_feature = "Name"
    target = "Survived"
    df_train = ds.read_file(
        file_name=file_train,
        nrows=10
    )
    X_train = df_train[features_two]
    y = df_train[target]
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    cross_validation_score = cross_val_score(
        estimator=logistic_regression,
        X=X_train,
        y=y,
        cv=3,
        scoring="accuracy"
    ).mean()
    print("Cross-validation score:", cross_validation_score)
    print()
    logistic_regression.fit(X=X_train, y=y)
    df_test = ds.read_file(
        file_name=file_new,
        nrows=10
    )
    X_test = df_test[features_two]
    predictions = logistic_regression.predict(X=X_test)
    print("Predictions:", predictions)
    print()
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit_transform(X=df_train[one_hot_encoder_features])
    # Now use ColumnTransformer and Pipeline on four features
    X_train = df_train[features]
    one_hot_encoder = OneHotEncoder()
    column_transformer = make_column_transformer(
        (one_hot_encoder, one_hot_encoder_features),
        ("passthrough", passthrough_features)
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X_train, y=y)
    X_test = df_test[features]
    predictions = pipeline.predict(X=X_test)
    print("Predictions X_test:", predictions)
    print()
    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(df_train[text_column])
    features = ["Parch", "Fare", "Embarked", "Sex", "Name"]
    X_train = df_train[features]
    X_test = df_test[features]
    column_transformer = make_column_transformer(
        (one_hot_encoder, one_hot_encoder_features),
        (vectorizer, vectorizer_feature),
        ("passthrough", passthrough_features)
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X_train, y=y)
    predictions = pipeline.predict(X=X_test)
    print("Predictions X_test:", predictions)
    print()


if __name__ == "__main__":
    main()
