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
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import datasense as ds


def main():
    start_time = time.perf_counter()
    features = ["Parch", "Fare", "Embarked", "Sex", "Name", "Age"]
    one_hot_encoder_features = ["Embarked", "Sex"]
    output_url = "master_machine_learning.html"
    header_title = "Master Machine Learning"
    header_id = "master-machine-learning"
    imputer_features = ["Age", "Fare"]
    passthrough_features = ["Parch"]
    file_data = "titanic_data.csv"
    file_new = "titanic_new.csv"
    vectorizer_features = "Name"
    target = "Survived"
    original_stdout = ds.html_begin(
        output_url=output_url,
        header_title=header_title,
        header_id=header_id
    )
    ds.script_summary(
        script_path=Path(__file__),
        action="started at"
    )
    df = ds.read_file(
        file_name=file_data,
        # nrows=10
    )
    ds.dataframe_info(
        df=df,
        file_in=file_data
    )
    X = df[features]
    y = df[target]
    df_new = ds.read_file(
        file_name=file_new,
        # nrows=10
    )
    ds.dataframe_info(
        df=df_new,
        file_in=file_new
    )
    X_new = df_new[features]
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
    one_hot_encoder = OneHotEncoder()
    vectorizer = CountVectorizer()
    imputer = SimpleImputer()
    imputer_constant = SimpleImputer(
        strategy="constant",
        fill_value="missing"
    )
    pipeline_imputer_ohe = make_pipeline(imputer_constant, one_hot_encoder)
    column_transformer = make_column_transformer(
        (pipeline_imputer_ohe, one_hot_encoder_features),
        (vectorizer, vectorizer_features),
        (imputer, imputer_features),
        ("passthrough", passthrough_features)
    )
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X, y=y)
    pipeline.predict(X=X_new)
    print("pipeline.predict(X=X_new)")
    print(pipeline.predict(X=X_new))
    print()
    # The following code throws an error
    # print(
    #     "Feature names",
    #     pipeline[:-1].get_feature_names_out()
    # )
    # print()
    print(pipeline)
    print()
    stop_time = time.perf_counter()
    ds.script_summary(script_path=Path(__file__), action="finished at")
    ds.report_summary(start_time=start_time, stop_time=stop_time)
    ds.html_end(original_stdout=original_stdout, output_url=output_url)


if __name__ == "__main__":
    main()
