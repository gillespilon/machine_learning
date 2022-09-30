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
import joblib


def main():
    features = ["Parch", "Fare", "Embarked", "Sex", "Name", "Age"]
    one_hot_encoder_features = ["Embarked", "Sex"]
    passthrough_features = ["Parch"]
    titanic_joblib = Path("pipeline.joblib")
    file_data = Path("titanic_data.csv")
    file_new = Path("titanic_new.csv")
    imputer_feature = ["Age", "Fare"]
    vectorizer_feature = "Name"
    target = "Survived"
    df = ds.read_file(
        file_name=file_data
    )
    ds.dataframe_info(
        df=df,
        file_in=file_data
    )
    # above showed that Age, Embarked had missing values
    X = df[features]
    y = df[target]
    df_new = ds.read_file(
        file_name=file_new
    )
    ds.dataframe_info(
        df=df_new,
        file_in=file_data
    )
    # above shows that Age, Fare had missing values
    X_new = df_new[features]
    # impute for NaN in Embarked, Sex before one-hot encoding
    imputer_constant = SimpleImputer(strategy="constant", fill_value="missing")
    one_hot_encoder = OneHotEncoder()
    # create a pipeline of two transformers for Embarked, Sex
    imputer_one_hot_encoder = make_pipeline(imputer_constant, one_hot_encoder)
    vectorizer = CountVectorizer()
    # impute for NaN in numeric columns Age, Fare
    imputer = SimpleImputer()
    # create a transformer that also includes the pipeline of transformers
    column_transformer = make_column_transformer(
        (imputer_one_hot_encoder, one_hot_encoder_features),
        (vectorizer, vectorizer_feature),
        (imputer, imputer_feature),
        ("passthrough", passthrough_features)
    )
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X, y=y)
    pipeline.predict(X=X_new)
    cross_validation_score = cross_val_score(
        estimator=pipeline,
        X=X,
        y=y,
        cv=5,
        scoring="accuracy"
    ).mean()
    print("Cross-validation score:", cross_validation_score)
    print()
    # Save the model to a joblib file
    joblib.dump(
        value=pipeline,
        filename=titanic_joblib
    )


if __name__ == "__main__":
    main()
