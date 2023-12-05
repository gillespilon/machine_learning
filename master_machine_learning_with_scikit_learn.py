#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Master machine learning with scikit-learn."
"""

from pathlib import Path
import time

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
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
    print("df.info():")
    print(df.info())
    print()
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
    print("df_new.info():")
    print(df_new.info())
    print()
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
        penalty="l2",
        C=1.0,
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
    # The following code throws an error
    # print(
    #     "Feature names",
    #     pipeline[:-1].get_feature_names_out()
    # )
    # print()
    # print(pipeline)
    # print()
    print("cross validation score, no tuning:")
    cross_validation_score = cross_val_score(
        estimator=pipeline,
        X=X,
        y=y,
        scoring="accuracy",
        cv=5,
    ).mean()
    print(cross_validation_score)
    print()
    # print("tune the logistic regression step")
    # print()
    # # Create a dictionary for GridSearchCV.
    # gridsearchcv_parameters = {}
    # gridsearchcv_parameters["logisticregression__penalty"] = ["l1", "l2"]
    # gridsearchcv_parameters["logisticregression__C"] = [0.1, 1.0, 10.0]
    # ds.print_dictionary_by_key(
    #     dictionary_to_print=gridsearchcv_parameters,
    #     title="parameters to tune for the logistic regression step:"
    # )
    # print()
    # gridsearchcv = GridSearchCV(
    #     estimator=pipeline,
    #     param_grid=gridsearchcv_parameters,
    #     scoring="accuracy",
    #     cv=5
    # )
    # gridsearchcv_fitted = gridsearchcv.fit(
    #     X=X,
    #     y=y
    # )
    # gridsearchcv_fitted_df = (
    #     pd.DataFrame(data=gridsearchcv_fitted.cv_results_)
    #     .sort_values("rank_test_score")
    # )
    # print("grid search fitted results")
    # print(gridsearchcv_fitted_df)
    # print()
    # print("gridsearchcv_fitted_df columns:")
    # print(gridsearchcv_fitted_df.columns)
    # print(
    #     "optimal values of "
    #     "logisticregression__penalty, "
    #     "logisticregression__C:"
    # )
    # print(gridsearchcv_fitted_df[[
    #     "rank_test_score",
    #     "mean_test_score",
    #     "param_logisticregression__C",
    #     "param_logisticregression__penalty"
    # ]])
    # print()
    # print("best score:")
    # print(gridsearchcv_fitted.best_score_)
    # print()
    # ds.print_dictionary_by_key(
    #     dictionary_to_print=gridsearchcv_fitted.best_params_,
    #     title="best parameters:"
    # )
    # print()
    print("tune the logistic regression step and the transformers")
    print("======================================================")
    print()
    print("names of the pipeline steps from the named_steps attribute:")
    print(pipeline.named_steps.keys())
    print()
    pipeline_parameters = list(pipeline.get_params().keys())
    ds.print_list_by_item(
        list_to_print=pipeline_parameters,
        title="pipeline parameters:"
    )
    print()
    print("names of the transformer names:")
    print(pipeline.named_steps["columntransformer"].named_transformers_)
    print()
    gridsearchcv_parameters = {}
    gridsearchcv_parameters["logisticregression__penalty"] = ["l1", "l2"]
    gridsearchcv_parameters["logisticregression__C"] = [0.1, 1.0, 10.0]
    gridsearchcv_parameters[
        "columntransformer__pipeline__onehotencoder__drop"
    ] = [None, "first"]
    gridsearchcv_parameters[
        "columntransformer__countvectorizer__ngram_range"
    ] = [(1, 1), (1, 2)]
    gridsearchcv_parameters[
        "columntransformer__simpleimputer__add_indicator"
    ] = [False, True]
    ds.print_dictionary_by_key(
        dictionary_to_print=gridsearchcv_parameters,
        title=(
            "parameters to tune for the logistic regression step and the "
            "transformers:"
        )
    )
    print()
    gridsearchcv = GridSearchCV(
        estimator=pipeline,
        param_grid=gridsearchcv_parameters,
        scoring="accuracy",
        n_jobs=-1,
        cv=5,
        verbose=1
    )
    gridsearchcv_fitted = gridsearchcv.fit(
        X=X,
        y=y
    )
    # gridsearchcv_fitted_df = (
    #     pd.DataFrame(data=gridsearchcv_fitted.cv_results_)
    #     .sort_values("rank_test_score")
    # )
    # print("grid search fitted results")
    # print(gridsearchcv_fitted_df)
    # print()
    # print("gridsearchcv_fitted_df columns:")
    # print(gridsearchcv_fitted_df.columns)
    # print(
    #     "optimal values of "
    #     "columntransformer__countvectorizer__ngram_range, "
    #     "columntransformer__pipeline__onehotencoder__drop, "
    #     "columntransformer__simpleimputer__add_indicator, "
    #     "logisticregression__penalty, "
    #     "logisticregression__C:")
    # print(gridsearchcv_fitted_df[[
    #     "rank_test_score",
    #     "mean_test_score",
    #     "param_columntransformer__countvectorizer__ngram_range",
    #     "param_columntransformer__pipeline__onehotencoder__drop",
    #     "param_columntransformer__simpleimputer__add_indicator",
    #     "param_logisticregression__C",
    #     "param_logisticregression__penalty"
    # ]])
    # print()
    print("best score:")
    print(gridsearchcv_fitted.best_score_)
    print()
    ds.print_dictionary_by_key(
        dictionary_to_print=gridsearchcv_fitted.best_params_,
        title="best parameters:"
    )
    print()
    print("best estimator (pipeline):")
    print(gridsearchcv_fitted.best_estimator_)
    print()
    print("use the best pipeline to make predictions:")
    print(gridsearchcv_fitted.predict(X=X_new))
    print()
    # Save the model to a joblib file
    joblib.dump(
        # value=pipeline,
        value=gridsearchcv_fitted.best_estimator_,
        filename=MASTER_ML_JOBLIB
    )
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
