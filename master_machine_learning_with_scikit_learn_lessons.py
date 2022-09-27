#! /usr/bin/env python3
"""
Lesson code for Kevin Markham's "Master machine learning with scikit-learn."
"""

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
    file_train = "titanic_train.csv"
    file_new = "titanic_new.csv"
    target = "Survived"
    print("master_machine_learning_with_scikit_learn_lessons.py")
    print()
    print("installed scikit-learn version:", sklearn.__version__)
    print("installed pandas version:      ", pd.__version__)
    print()
    print("2.1 Loading and exploring a dataset")
    print()
    df_train = ds.read_file(
        file_name=file_train,
        nrows=10
    )
    print("Create df_train:")
    print()
    print(df_train)
    # Use intuition to select two features
    # Create the X_train DataFrame
    X_train = df_train[features_two]
    print()
    print("Create X_train DataFrame:")
    print(X_train)
    print()
    # Create the y Series
    y = df_train[target]
    print("Create y Series for multiclass target:")
    print(y)
    print()
    # Check the shapes of X_train, y
    print("X_train shape:", X_train.shape)
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
        X=X_train,
        y=y,
        cv=3,
        scoring="accuracy"
    ).mean()
    print("Cross-validation score:", cross_validation_score)
    print()
    print("2.3 Using the model to make predictions")
    print()
    logistic_regression.fit(X=X_train, y=y)
    df_test = ds.read_file(
        file_name=file_new,
        nrows=10
    )
    print("Create df_test:")
    print()
    print(df_test)
    print()
    X_test = df_test[features_two]
    print("Create X_test:")
    print()
    print(X_test)
    print()
    logistic_regression_predict = logistic_regression.predict(X=X_test)
    print("logistic_regression.predict:")
    print(logistic_regression_predict)
    print()
    print("2.7 Add model's predictions to the X_test DataFrame")
    print()
    predictions = pd.Series(
        data=logistic_regression_predict, index=X_test.index, name='Prediction'
    )
    X_test_predictions = pd.concat(objs=[X_test, predictions], axis='columns')
    print("Create X_test with predictions:")
    print()
    print(X_test_predictions)
    print()
    print("2.8 Determine the confidence level of each prediction")
    print()
    print("Probabilities of 0, 1:")
    print()
    print(logistic_regression.predict_proba(X=X_test))
    print()
    print("Probabilities of 1:")
    print()
    print(logistic_regression.predict_proba(X=X_test)[:, 1])
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
    #     X=X_train,
    #     y=y,
    #     cv=kf,
    #     scoring = "accuracy"
    # )
    print("3.1 Introduction to one-hot encoding")
    print("3.3 One-hot encoding of multiple features")
    print()
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit_transform(X=df_train[["Embarked", "Sex"]])
    print("Embarked, Sex categories:")
    print()
    print(one_hot_encoder.categories_)
    print()
    print("4. Improving your workflow with ColumnTransformer and Pipeline")
    print("4.1 Preprocessing features with ColumnTransformer")
    print()
    X_train = df_train[features_four]
    one_hot_encoder = OneHotEncoder()
    column_transformer = make_column_transformer(
        (one_hot_encoder, ["Embarked", "Sex"]),
        remainder="passthrough"
    )
    column_transformer.fit_transform(X=X_train)
    print("Features names of X_train:")
    print()
    print(column_transformer.get_feature_names_out())
    print()
    print("4.2 Chaining steps with Pipeline")
    print()
    pipeline = make_pipeline(column_transformer, logistic_regression)
    pipeline.fit(X=X_train, y=y)
    print("4.3 Using the Pipeline to make predictions")
    print()
    X_test = df_test[features_four]
    pipeline.predict(X=X_test)
    print("pipeline.predict(X=X_test):", pipeline.predict(X=X_test))
    print()
    print(pipeline.named_steps.keys())
    print()
    print("pipeline:")
    print()
    set_config(display="text")
    print(pipeline)
    print()


if __name__ == "__main__":
    main()
