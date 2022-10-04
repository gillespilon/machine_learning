#! /usr/bin/env python3
"""
Code to mask outliers with a function, column transformer, and pipeline

The goal is to mask outliers and then process these within scikit-learn,
not within pandas, in order to avoid data leakage.

./outliers_masking_pipeline_scikit_learn.py >
    outliers_masking_pipeline_scikit_learn.txt
"""

from pathlib import Path
import time

import datasense as ds
import pandas as pd


def main():
    file_name = "outliers_missing.csv"
    features = ["X"]
    target = "Y"
    print("outliers_masking_pipeline_scikit_learn.py")
    print()
    # create a dataset
    df = ds.read_file(file_name=file_name)
    ds.dataframe_info(
        df=df,
        file_in=file_name
    )
    mask_values = [("X", -4, 4)]
    # Replace outliers with NaN
    for column, lowvalue, highvalue in mask_values:
        df[column] = df[column].mask(
            cond=(df[column] <= lowvalue) | (df[column] >= highvalue),
            other=pd.NA
        )
    ds.dataframe_info(
        df=df,
        file_in=file_name
    )


if __name__ == "__main__":
    main()
