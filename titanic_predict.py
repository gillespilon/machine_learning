#! /usr/bin/env python3
"""
Python script to load a model and make predictions on new data
"""

from pathlib import Path

import datasense as ds
import pandas as pd
import joblib


def main():
    features = ["Parch", "Fare", "Embarked", "Sex", "Name", "Age"]
    file_predictions = "titanic_predictions.csv"
    titanic_joblib = Path("pipeline.joblib")
    file_test = "titanic_new.csv"
    pipeline_from_joblib = joblib.load(filename=titanic_joblib)
    # df_test = pd.read_csv(
    #     filepath_or_buffer=file_test
    # )
    df_test = ds.read_file(
        file_name=file_test
    )
    X_test = df_test[features]
    predictions_ndarray = pipeline_from_joblib.predict(X=X_test)
    predictions_series = pd.Series(
        data=predictions_ndarray,
        index=X_test.index,
        name='Survived prediction'
    )
    X_test_predictions = pd.concat(
        objs=[X_test, predictions_series],
        axis='columns'
    )
    # X_test_predictions.to_csv(
    #     path_or_buf=file_predictions
    # )
    ds.save_file(
        df=X_test_predictions,
        file_name=file_predictions
    )


if __name__ == "__main__":
    main()
