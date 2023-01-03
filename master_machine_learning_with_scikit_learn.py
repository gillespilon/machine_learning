#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Master machine learning with scikit-learn."
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import datasense as ds


def main():
    features = ["Parch", "Fare", "Embarked", "Sex", "Name", "Age"]
    one_hot_encoder_features = ["Embarked", "Sex"]
    passthrough_features = ["Parch", "Fare"]
    file_data = "titanic_data.csv"
    file_new = "titanic_new.csv"
    vectorizer_features = "Name"
    imputer_features = ["Age"]
    target = "Survived"
    print("master_machine_learning_with_scikit_learn.py")
    print()
    df = ds.read_file(
        file_name=file_data,
        nrows=10
    )
    X = df[features]
    y = df[target]
    df_new = ds.read_file(
        file_name=file_new,
        nrows=10
    )
    X_new = df_new[features]
    one_hot_encoder = OneHotEncoder()
    vectorizer = CountVectorizer()
    imputer = SimpleImputer()
    # option 1
    column_transformer = make_column_transformer(
        (one_hot_encoder, one_hot_encoder_features),
        (vectorizer, vectorizer_features),
        (imputer, imputer_features),
        ("passthrough", passthrough_features)
    )
    # option 2
    # column_transformer = make_column_transformer(
    #     (one_hot_encoder, one_hot_encorder_features),
    #     remainder="passthrough"
    # )
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X, y=y)
    pipeline.predict(X=X_new)
    print("pipeline.predict(X=X_new):", pipeline.predict(X=X_new))
    print()


if __name__ == "__main__":
    main()
