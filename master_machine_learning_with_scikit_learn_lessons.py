#! /usr/bin/env python3
"""
Lesson code for Kevin Markham's "Master machine learning with scikit-learn."
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn import set_config
import datasense as ds
import pandas as pd
import sklearn


def main():
    features_four = ["Parch", "Fare", "Embarked", "Sex"]
    features_two = ["Parch", "Fare"]
    file_data = "titanic_data.csv"
    file_new = "titanic_new.csv"
    feature_name = "Name"
    target = "Survived"
    print("master_machine_learning_with_scikit_learn_lessons.py")
    print()
    print("installed scikit-learn version:", sklearn.__version__)
    print("installed pandas version:      ", pd.__version__)
    print()
    print("2.1 Loading and exploring a dataset")
    print()
    df = ds.read_file(
        file_name=file_data,
        nrows=10
    )
    print("Create df:")
    print()
    print(df)
    # Use intuition to select two features
    # Create the X DataFrame
    X = df[features_two]
    print()
    print("Create X DataFrame:")
    print(X)
    print()
    # Create the y Series
    y = df[target]
    print("Create y Series for multiclass target:")
    print(y)
    print()
    # Check the shapes of X, y
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()
    print("2.2 Building and evaluate a model")
    print()
    print("Create model, evaluation metric for multiclass classification")
    print()
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    cross_validation_score = cross_val_score(
        estimator=logistic_regression,
        X=X,
        y=y,
        cv=3,
        scoring="accuracy"
    ).mean()
    print("Cross-validation score:", cross_validation_score)
    print()
    print("2.3 Using the model to make predictions")
    print()
    logistic_regression.fit(X=X, y=y)
    df_new = ds.read_file(
        file_name=file_new,
        nrows=10
    )
    print("Create df_new:")
    print()
    print(df_new)
    print()
    X_new = df_new[features_two]
    print("Create X_new:")
    print()
    print(X_new)
    print()
    logistic_regression_predict = logistic_regression.predict(X=X_new)
    print("logistic_regression.predict:")
    print(logistic_regression_predict)
    print()
    print("2.7 Add model's predictions to the X_new DataFrame")
    print()
    predictions = pd.Series(
        data=logistic_regression_predict, index=X_new.index, name='Prediction'
    )
    X_new_predictions = pd.concat(objs=[X_new, predictions], axis='columns')
    print("Create X_new with predictions:")
    print()
    print(X_new_predictions)
    print()
    print("2.8 Determine the confidence level of each prediction")
    print()
    print("Probabilities of 0, 1:")
    print()
    print(logistic_regression.predict_proba(X=X_new))
    print()
    print("Probabilities of 1:")
    print()
    print(logistic_regression.predict_proba(X=X_new)[:, 1])
    print()
    print("2.10 Show all of the model parameters")
    print()
    print("logistic_regression parameters:")
    print()
    print(logistic_regression.get_params())
    print()
    # 2.12 If you need to shuffle the samples when using cross-validation
    # Used when samples are ordered and shuffling is needed
    # from sklearn.model_selection import StratifiedKFold
    # kf = StratifiedKFold(3, shuffle=True, random_state=1)
    # cross_val_score(
    #     estimate=logistic_regression,
    #     X=X,
    #     y=y,
    #     cv=kf,
    #     scoring = "accuracy"
    # )
    print("3.1 Introduction to one-hot encoding")
    print("3.3 One-hot encoding of multiple features")
    print()
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit_transform(X=df[["Embarked", "Sex"]])
    print("Embarked, Sex categories:")
    print()
    print(one_hot_encoder.categories_)
    print()
    print("4. Improving your workflow with ColumnTransformer and Pipeline")
    print("4.1 Preprocessing features with ColumnTransformer")
    print()
    X = df[features_four]
    one_hot_encoder = OneHotEncoder()
    column_transformer = make_column_transformer(
        (one_hot_encoder, ["Embarked", "Sex"]),
        remainder="passthrough"
    )
    column_transformer.fit_transform(X=X)
    print("Features names of X:")
    print()
    print(column_transformer.get_feature_names_out())
    print()
    print("4.2 Chaining steps with Pipeline")
    print()
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X, y=y)
    print("4.3 Using the Pipeline to make predictions")
    print()
    X_new = df_new[features_four]
    pipeline.predict(X=X_new)
    print("pipeline.predict(X=X_new):", pipeline.predict(X=X_new))
    print()
    print(pipeline.named_steps.keys())
    print()
    print("pipeline:")
    print()
    set_config(display="text")
    print(pipeline)
    print()
    print("6. Encoding text data")
    print("6.1 Vectorizing text")
    df = ds.read_file(
        file_name=file_data,
        nrows=10
    )
    y = df[target]
    df_new = ds.read_file(
        file_name=file_new,
        nrows=10
    )
    ohe = OneHotEncoder()
    logistic_regression = LogisticRegression(
        solver="liblinear",
        random_state=1
    )
    set_config(display="diagram")
    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(df[feature_name])
    print("document_term_matrix")
    print(document_term_matrix)
    print()
    print("feature names:")
    print(vectorizer.get_feature_names_out())
    print()
    print(
        pd.DataFrame(
            data=document_term_matrix.toarray(),
            columns=vectorizer.get_feature_names_out())
    )
    features_five = ["Parch", "Fare", "Embarked", "Sex", "Name"]
    X = df[features_five]
    column_transformer = make_column_transformer(
        (ohe, ["Embarked", "Sex"]),
        (vectorizer, "Name"),
        ("passthrough", ["Parch", "Fare"])
    )
    column_transformer.fit_transform(X)
    print(column_transformer.get_feature_names_out())
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X, y=y)
    X_new = df_new[features_five]
    pipeline.predict(X=X_new)
    print("pipeline.predict(X=X_new):", pipeline.predict(X=X_new))


if __name__ == "__main__":
    main()
