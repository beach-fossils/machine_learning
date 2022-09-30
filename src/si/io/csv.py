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

    if features and label:
        df = pd.read_csv(filename, sep=sep)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        features = df.columns[:-1].values
        label = df.columns[-1]
        return Dataset(x, y, features, label)
    elif features:
        df = pd.read_csv(filename, sep=sep)
        x = df.iloc[:, :].values
        features = df.columns[:].values
        return Dataset(x, features=features)
    elif label:
        df = pd.read_csv(filename, sep=sep)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        label = df.columns[-1]
        return Dataset(x, y, label=label)
    else:
        df = pd.read_csv(filename, sep=sep)
        x = df.iloc[:, :].values
        return Dataset(x)


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
        df = pd.DataFrame(dataset.x, columns=dataset.features)
        df[dataset.label] = dataset.y
        df.to_csv(filename, sep=sep, index=False)
    elif features:
        df = pd.DataFrame(dataset.x, columns=dataset.features)
        df.to_csv(filename, sep=sep, index=False)
    elif label:
        df = pd.DataFrame(dataset.x)
        df[dataset.label] = dataset.y
        df.to_csv(filename, sep=sep, index=False)
    else:
        df = pd.DataFrame(dataset.x)
        df.to_csv(filename, sep=sep, index=False)


if __name__ == '__main__':
    # Test the function read_csv
    dataset = read_csv('/Users/josediogomoura/machine_learning/datasets/datasets/iris.csv', features=True, label=True)
    print(dataset.summary())

    # Test the function write_csv
    write_csv(dataset, '/Users/josediogomoura/machine_learning/datasets/datasets/iris2.csv', features=True, label=True)

