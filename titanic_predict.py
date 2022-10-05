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
    file_predictions = Path("titanic_predictions.csv")
    titanic_joblib = Path("pipeline.joblib")
    file_new = Path("titanic_new.csv")
    pipeline_from_joblib = joblib.load(filename=titanic_joblib)
    # df_new = pd.read_csv(
    #     filepath_or_buffer=file_new
    # )
    df_new = ds.read_file(
        file_name=file_new
    )
    X_new = df_new[features]
    predictions_ndarray = pipeline_from_joblib.predict(X=X_new)
    predictions_series = pd.Series(
        data=predictions_ndarray,
        index=X_new.index,
        name='Survived prediction'
    )
    X_new_predictions = pd.concat(
        objs=[X_new, predictions_series],
        axis='columns'
    )
    # X_new_predictions.to_csv(
    #     path_or_buf=file_predictions
    # )
    ds.save_file(
        df=X_new_predictions,
        file_name=file_predictions
    )


if __name__ == "__main__":
    main()
