#! /usr/bin/env python3
"""
Python script to load a model and make predictions on new data
"""

from pathlib import Path

import datasense as ds
import joblib


def main():
    features = ["Parch", "Fare", "Embarked", "Sex", "Name"]
    titanic_joblib = Path("pipeline.joblib")
    file_test = "titanic_new.csv"
    pipeline_from_joblib = joblib.load(filename=titanic_joblib)
    df_test = ds.read_file(
        file_name=file_test,
        nrows=10
    )
    X_test = df_test[features]
    predictions = pipeline_from_joblib.predict(X=X_test)
    print("Predictions X_test:", predictions)
    print()


if __name__ == "__main__":
    main()
