#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Building an effective machine learning
workflow with scikit-learn"
"""

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import datasense as ds
import joblib


def main():
    features = ["Parch", "Fare", "Embarked", "Sex", "Name"]
    one_hot_encoder_features = ["Embarked", "Sex"]
    passthrough_features = ["Parch", "Fare"]
    titanic_joblib = Path("pipeline.joblib")
    file_train = Path("titanic_train.csv")
    file_test = Path("titanic_new.csv")
    vectorizer_feature = "Name"
    target = "Survived"
    df_train = ds.read_file(
        file_name=file_train,
        nrows=10
    )
    X_train = df_train[features]
    y = df_train[target]
    df_test = ds.read_file(
        file_name=file_test,
        nrows=10
    )
    X_test = df_test[features]
    # Use ColumnTransformer and Pipeline
    one_hot_encoder = OneHotEncoder()
    vectorizer = CountVectorizer()
    column_transformer = make_column_transformer(
        (one_hot_encoder, one_hot_encoder_features),
        (vectorizer, vectorizer_feature),
        ("passthrough", passthrough_features)
    )
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X_train, y=y)
    predictions = pipeline.predict(X=X_test)
    print("Predictions X_test:", predictions)
    print()
    # Save the model to a joblib file
    joblib.dump(
        value=pipeline,
        filename=titanic_joblib
    )


if __name__ == "__main__":
    main()
