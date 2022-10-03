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
    # the next line is unnecessary because I use titanic_predict.py
    X_new = df_new[features]
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
        (imputer_one_hot_encoder, one_hot_encoder_features),
        (vectorizer, vectorizer_feature),
        (imputer, imputer_feature),
        ("passthrough", passthrough_features)
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
        filename=titanic_joblib
    )


if __name__ == "__main__":
    main()
