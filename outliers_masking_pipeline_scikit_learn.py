#! /usr/bin/env python3
"""
Code to mask outliers with a function, column transformer, and pipeline

The goal is to mask outliers and then process these within scikit-learn,
not within pandas, in order to avoid data leakage.

./outliers_masking_pipeline_scikit_learn.py >
    outliers_masking_pipeline_scikit_learn.txt
"""

from typing import List, Tuple
from pathlib import Path

import datasense as ds
import pandas as pd


def mask_outliers(
    *,
    df: pd.DataFrame,
    maskvalues: List[Tuple[str, float, float]]
) -> pd.DataFrame:
    for feature, lowvalue, highvalue in maskvalues:
        df[feature] = df[feature].mask(
            cond=(df[feature] <= lowvalue) | (df[feature] >= highvalue),
            other=pd.NA
        )
    return pd.DataFrame(data=df)


def main():
    file_name = Path("outliers_missing.csv")
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
    maskvalues = [("X", -4, 4)]
    df = mask_outliers(
        df=df,
        maskvalues=maskvalues
    )
    ds.dataframe_info(
        df=df,
        file_in=file_name
    )


if __name__ == "__main__":
    main()
