import pandas as pd
import numpy as np
from si.data.dataset import Dataset


def read_csv(filename, sep=',', features=False, label=False):
    """Reads a csv file and returns a Dataset object
    parameters:
    filename: The name of the csv file.
    sep: The separator used in the csv file.
    features: boolean indicating if there is a feature row or not.
    label: boolean indicating if there is a label row or not.
    returns:
    A Dataset object."""

    if features:
        df = pd.read_csv(filename, sep=sep)
        features = df.columns[:-1]
        label = df.columns[-1]
        X = df[features].values
        y = df[label].values
        return Dataset(X, y, features, label)
    elif label:
        df = pd.read_csv(filename, sep=sep)
        label = df.columns[-1]
        X = df.values
        y = df[label].values
        features = None
        return Dataset(X, y, features=None, label=label)
    else:
        df = pd.read_csv(filename, sep=sep)
        X = df.values
        y = None
        label_name = None
        return Dataset(X, y, label=label_name)


def write_csv(dataset, filename, sep=',', features=False, label=False):
    """Writes a Dataset object to a csv file.
    parameters:
    dataset: A Dataset object.
    filename: The name of the csv file.
    sep: The separator used in the csv file.
    features: boolean indicating if there is a feature row or not.
    label: boolean indicating if there is a label row or not.
    returns:
    A Dataset object."""

    if features and label:
        df = pd.DataFrame(dataset.X, columns=dataset.features)
        df[dataset.label] = dataset.y
        df.to_csv(filename, sep=sep, index=False)
    elif features:
        df = pd.DataFrame(dataset.X, columns=dataset.features)
        df.to_csv(filename, sep=sep, index=False)
    elif label:
        df = pd.DataFrame(dataset.X)
        df[dataset.label] = dataset.y
        df.to_csv(filename, sep=sep, index=False)
    else:
        df = pd.DataFrame(dataset.X)
        df.to_csv(filename, sep=sep, index=False)


if __name__ == '__main__':
    # Test the function read_csv
    dataset = read_csv('/Users/josediogomoura/machine_learning/datasets/datasets/iris.csv', sep=',', features=True,
                       label=True)
    print(dataset.summary())

    # Test the function write_csv
    # write_csv(dataset, '/Users/josediogomoura/machine_learning/datasets/datasets/iris2.csv', features=True, label=True)
