#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Building an effective machine learning
workflow with scikit-learn"
"""

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import datasense as ds
import joblib


def main():
    FEATURES = ["Parch", "Fare", "Embarked", "Sex", "Name", "Age"]
    ONE_HOT_ENCODER_FEATURES = ["Embarked", "Sex"]
    TITANIC_JOBLIB = Path("pipeline.joblib")
    FILE_DATA = Path("titanic_data.csv")
    file_new = Path("titanic_new.csv")
    IMPUTER_FEATURE = ["Age", "Fare"]
    PASSTHROUGH_FEATURES = ["Parch"]
    VECTORIZER_FEATURE = "Name"
    TARGET = "Survived"
    df = ds.read_file(
        file_name=FILE_DATA
    )
    ds.dataframe_info(
        df=df,
        file_in=FILE_DATA
    )
    # above showed that Age, Embarked had missing values
    X = df[FEATURES]
    y = df[TARGET]
    df_new = ds.read_file(
        file_name=file_new
    )
    ds.dataframe_info(
        df=df_new,
        file_in=FILE_DATA
    )
    # above shows that Age, Fare had missing values
    # the next line is unnecessary because I use titanic_predict.py
    X_new = df_new[FEATURES]
    # impute for NaN in Embarked, Sex before one-hot encoding
    imputer_constant = SimpleImputer(strategy="constant", fill_value="missing")
    one_hot_encoder = OneHotEncoder()
    # create a pipeline of two transformers for Embarked, Sex
    imputer_one_hot_encoder = make_pipeline(imputer_constant, one_hot_encoder)
    vectorizer = CountVectorizer()
    # impute for NaN in numeric columns Age, Fare
    imputer = SimpleImputer()
    # create a transformer that also includes the pipeline of two transformers
    column_transformer = make_column_transformer(
        (imputer_one_hot_encoder, ONE_HOT_ENCODER_FEATURES),
        (vectorizer, VECTORIZER_FEATURE),
        (imputer, IMPUTER_FEATURE),
        ("passthrough", PASSTHROUGH_FEATURES)
    )
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    logistic_regression_selection = LogisticRegression(
        solver="liblinear",
        penalty="l1",
        random_state=1
    )
    selection = SelectFromModel(
        estimator=logistic_regression_selection,
        threshold="mean"
    )
    pipeline = make_pipeline(
        column_transformer, selection, logistic_regression
    )
    pipeline.fit(X=X, y=y)
    # tuning hyperparameters
    # what are the steps of the pipeline?
    print("Steps of the pipeline:", pipeline.named_steps.keys())
    print()
    # what are the transformers in columntransformer?
    print("Transformers in columntransformer:")
    print(pipeline.named_steps.columntransformer.named_transformers_)
    print()
    params = {}
    params["logisticregression__penalty"] = ["l1", "l2"]
    params["logisticregression__C"] = [0.1, 1, 10]
    params["columntransformer__pipeline__onehotencoder__drop"] =\
        [None, "first"]
    params["columntransformer__countvectorizer__ngram_range"] =\
        [(1, 1), (1, 2)]
    params["columntransformer__simpleimputer__add_indicator"] = [False, True]
    params["selectfrommodel__threshold"] = ["mean", "median"]
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="accuracy",
        n_jobs=-1,
        cv=10
    )
    grid.fit(X=X, y=y)
    print("Best score:", grid.best_score_)
    print("Best hyperparameters:")
    print(grid.best_params_)
    print()
    # the next line is unnecessary because I use titanic_predict.py
    grid.predict(X=X_new)
    # Save the model to a joblib file
    joblib.dump(
        value=grid,
        filename=TITANIC_JOBLIB
    )


if __name__ == "__main__":
    main()
