#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Master machine learning with scikit-learn."
"""

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import datasense as ds


def main():
    features = ["Parch", "Fare", "Embarked", "Sex"]
    one_hot_encoder_features = ["Embarked", "Sex"]
    passthrough_features = ["Parch", "Fare"]
    file_data = "titanic_data.csv"
    file_new = "titanic_new.csv"
    target = "Survived"
    print("master_machine_learning_with_scikit_learn.py")
    print()
    df = ds.read_file(
        file_name=file_data,
        nrows=10
    )
    df = df.dropna(subset=[target])
    X = df[features]
    y = df[target]
    df_new = ds.read_file(
        file_name=file_new,
        nrows=10
    )
    X_new = df_new[features]
    one_hot_encoder = OneHotEncoder()
    # option 1
    column_transformer = make_column_transformer(
        (one_hot_encoder, one_hot_encoder_features),
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
