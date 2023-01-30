#! /usr/bin/env python3
"""
Python script to load a model and make predictions on new data
"""

from pathlib import Path

import datasense as ds
import pandas as pd
import joblib


def main():
    FEATURES = ["Parch", "Fare", "Embarked", "Sex", "Name", "Age"]
    FILE_PREDICTIONS = Path("titanic_predictions.csv")
    MASTER_ML_JOBLIB = Path("master_ml_pipeline.joblib")
    SERIES_NAME = "Survived prediction"
    FILE_NEW = Path("titanic_new.csv")
    pipeline_from_joblib = joblib.load(filename=MASTER_ML_JOBLIB)
    # df_new = pd.read_csv(
    #     filepath_or_buffer=FILE_NEW
    # )
    df_new = ds.read_file(
        file_name=FILE_NEW
    )
    X_new = df_new[FEATURES]
    predictions_ndarray = pipeline_from_joblib.predict(X=X_new)
    predictions_series = pd.Series(
        data=predictions_ndarray,
        index=X_new.index,
        name=SERIES_NAME
    )
    X_new_predictions = pd.concat(
        objs=[X_new, predictions_series],
        axis="columns"
    )
    # X_new_predictions.to_csv(
    #     path_or_buf=FILE_PREDICTIONS
    # )
    ds.save_file(
        df=X_new_predictions,
        file_name=FILE_PREDICTIONS
    )


if __name__ == "__main__":
    main()
