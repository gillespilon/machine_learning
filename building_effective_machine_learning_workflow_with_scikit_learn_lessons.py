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
from sklearn.impute import SimpleImputer
import datasense as ds
import pandas as pd
# import sklearn


def main():
    # print(sklearn.__version__)
    features = ["Parch", "Fare", "Embarked", "Sex"]
    one_hot_encoder_features = ["Embarked", "Sex"]
    imputer_constant_feature = ["Embarked"]
    passthrough_features = ["Parch"]
    imputer_feature = ["Age", "Fare"]
    features_two = ["Parch", "Fare"]
    # kaggle dataset
    file_data = Path("titanic_data.csv")
    # kaggle new dataset`
    file_new = Path("titanic_new.csv")
    vectorizer_feature = "Name"
    target = "Survived"
    df = ds.read_file(
        file_name=file_data,
        nrows=10
    )
    X_train = df[features_two]
    y = df[target]
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
    df_new = ds.read_file(
        file_name=file_new,
        nrows=10
    )
    X_new = df_new[features_two]
    predictions = logistic_regression.predict(X=X_new)
    print("Predictions:", predictions)
    print()
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit_transform(X=df[one_hot_encoder_features])
    # Now use ColumnTransformer and Pipeline on four features
    X_train = df[features]
    one_hot_encoder = OneHotEncoder()
    column_transformer = make_column_transformer(
        (one_hot_encoder, one_hot_encoder_features),
        ("passthrough", passthrough_features)
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X_train, y=y)
    X_new = df_new[features]
    predictions = pipeline.predict(X=X_new)
    print("Predictions X_new:", predictions)
    print()
    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(
        df[vectorizer_feature]
    )
    features = ["Parch", "Fare", "Embarked", "Sex", "Name", "Age"]
    passthrough_features = ["Parch"]
    X_train = df[features]
    X_new = df_new[features]
    imputer = SimpleImputer()
    column_transformer = make_column_transformer(
        (one_hot_encoder, one_hot_encoder_features),
        (vectorizer, vectorizer_feature),
        (imputer, imputer_feature),
        ("passthrough", passthrough_features)
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X_train, y=y)
    predictions = pipeline.predict(X=X_new)
    print("Predictions X_new:", predictions)
    print()
    # Now use the full dataset`
    df = ds.read_file(
        file_name=file_data
    )
    df_new = ds.read_file(
        file_name=file_new
    )
    print("Missing values in df?")
    print(df.isna().sum())
    print()
    print("Missing values in df_new?")
    print(df_new.isna().sum())
    print()
    X_train = df[features]
    y = df[target]
    X_new = df_new[features]
    imputer_constant = SimpleImputer(strategy="constant", fill_value="missing")
    # create a pipeline of two transformers
    # this solves the problem of Embarked having missing values
    imputer_one_hot_encoder = make_pipeline(imputer_constant, one_hot_encoder)
    column_transformer = make_column_transformer(
        (imputer_one_hot_encoder, one_hot_encoder_features),
        (vectorizer, vectorizer_feature),
        (imputer, imputer_feature),
        ("passthrough", passthrough_features)
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X_train, y=y)
    predictions = pipeline.predict(X=X_new)
    cross_validation_score = cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y,
        cv=5,
        scoring="accuracy"
    ).mean()
    print("Cross-validation score:", cross_validation_score)
    print()


if __name__ == "__main__":
    main()
