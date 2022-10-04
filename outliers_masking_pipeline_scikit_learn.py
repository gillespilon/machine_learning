#! /usr/bin/env python3
"""
Code to mask outliers with a function, column transformer, and pipeline
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
    df = ds.read_file(file_name=file_name)
    ds.dataframe_info(
        df=df,
        file_in=file_name
    )


if __name__ == "__main__":
    main()
