#! /usr/bin/env python3
"""
Machine learning of the iris dataset.
"""

from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    print(type(iris))
    print(iris.data)
    print(iris.feature_names)
    print(iris.target)
    print(iris.target_names)
    print(iris.data.shape)
    print(iris.target.shape)


if __name__ == "__main__":
    main()
