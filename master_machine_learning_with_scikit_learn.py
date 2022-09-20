#! /usr/bin/env python3
"""
Master machine learning with scikit-learn
"""

from pathlib import Path
import time

import datasense as ds
import pandas as pd
import sklearn


def main():
    print("installed scikit-learn version:", sklearn.__version__)
    print("installed pandas version:      ", pd.__version__)


if __name__ == "__main__":
    main()
