#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Master machine learning with scikit-learn."
"""

from pathlib import Path
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import CategoricalDtype
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import datasense as ds
import pandas as pd
import joblib


def main():
    start_time = time.perf_counter()
    FEATURES = ["Parch", "Fare", "Embarked", "Sex", "Name", "Age"]
    MASTER_ML_JOBLIB = Path("master_ml_pipeline.joblib")
    ONE_HOT_ENCODER_FEATURES = ["Embarked", "Sex"]
    OUTPUT_URL = "master_machine_learning.html"
    HEADER_TITLE = "Master Machine Learning"
    HEADER_ID = "master-machine-learning"
    IMPUTER_FEATURES = ["Age", "Fare"]
    PASSTHROUGH_FEATURES = ["Parch"]
    FILE_DATA = "titanic_data.csv"
    FILE_NEW = "titanic_new.csv"
    VECTORIZER_FEATURES = "Name"
    TARGET = "Survived"
    original_stdout = ds.html_begin(
        output_url=OUTPUT_URL,
        header_title=HEADER_TITLE,
        header_id=HEADER_ID
    )
    ds.script_summary(
        script_path=Path(__file__),
        action="started at"
    )
    # Read training data, define X and y
    df = ds.read_file(
        file_name=FILE_DATA,
        # nrows=10
    ).astype(dtype={"Embarked": CategoricalDtype()})
    print(df.info())
    ds.dataframe_info(
        df=df,
        file_in=FILE_DATA
    )
    X = df[FEATURES]
    y = df[TARGET]
    # the next line is unnecessary because I use master_ml_pipeline.py
    # Read new data, define X_new
    df_new = ds.read_file(
        file_name=FILE_NEW,
        # nrows=10
    ).astype(dtype={"Embarked": CategoricalDtype()})
    print(df.info())
    ds.dataframe_info(
        df=df_new,
        file_in=FILE_NEW
    )
    X_new = df_new[FEATURES]
    print("Missing data")
    print("============")
    print(
        "Considering the training data and the new date, "
        "Age, Cabin, and Fare have missing values. "
        "We will use a SimpleImputer for Age and Fare. "
        "We will use a Pipeline with a SimpleImputer and OneHotEncoder "
        "for Embarked. "
        "Cabin is not a Feature in our model."
    )
    print()
    # Create four instances of transformers
    imputer = SimpleImputer()
    imputer_constant = SimpleImputer(
        strategy="constant",
        fill_value="missing"
    )
    one_hot_encoder = OneHotEncoder()
    vectorizer = CountVectorizer()
    # Create two-step transformer pipeline of constant value imputation and
    # one-hot encoding
    pipeline_imputer_ohe = make_pipeline(imputer_constant, one_hot_encoder)
    # Create column transformer
    column_transformer = make_column_transformer(
        (pipeline_imputer_ohe, ONE_HOT_ENCODER_FEATURES),
        (vectorizer, VECTORIZER_FEATURES),
        (imputer, IMPUTER_FEATURES),
        ("passthrough", PASSTHROUGH_FEATURES)
    )
    # Create instance of logistic regression
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    # Create two-step pipeline, fit the pipeline, make predictions
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X, y=y)
    # the next line is unnecessary because I use master_ml_pipeline.py
    pipeline.predict(X=X_new)
    print("pipeline.predict(X=X_new)")
    predictions = pipeline.predict(X=X_new)
    print(predictions)
    print()
    # Save the model to a joblib file
    joblib.dump(
        value=pipeline,
        filename=MASTER_ML_JOBLIB
    )
    # The following code throws an error
    # print(
    #     "Feature names",
    #     pipeline[:-1].get_feature_names_out()
    # )
    # print()
    print(pipeline)
    print()
    stop_time = time.perf_counter()
    ds.script_summary(
        script_path=Path(__file__),
        action="finished at"
    )
    ds.report_summary(
        start_time=start_time,
        stop_time=stop_time
    )
    ds.html_end(
        original_stdout=original_stdout,
        output_url=OUTPUT_URL
    )


if __name__ == "__main__":
    main()
