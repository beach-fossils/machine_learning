#  numpy.genfromtxt
import pandas as pd
import numpy as np
from numpy import genfromtxt
from si.data.dataset import Dataset
from numpy import savetxt


def read_data_file(filename, sep=',', label=False):
    """Reads a generic file, transforms it into a .txt file and returns a Dataset object.
    parameters:
    filename: The name of the file.
    sep: The separator used in the file.
    label: boolean indicating if there is a label row or not.
    returns:
    A Dataset object."""

    # transform the generic file into a txt file

    # read the generic file

    if label:
        data = genfromtxt(filename, delimiter=sep)
        x = data[:, :-1]  # all rows, all columns except the last one (label)
        y = data[:, -1]  # all rows, only the last column (label)
        return Dataset(x, y, label=True)
    else:
        data = genfromtxt(filename, delimiter=sep)
        return Dataset(data, label=False)


def write_data_file(dataset, filename, sep=',', label=False):
    """Writes a Dataset object to a generic .txt file.
    parameters:
    dataset: A Dataset object.
    filename: The name of the file.
    sep: The separator used in the file.
    label: boolean indicating if there is a label row or not.
    returns:
    A Dataset object."""

    if label:
        x = dataset.x
        y = dataset.y
        data = np.append(x, y, axis=1)
        savetxt(filename, data, delimiter=sep, fmt='%f')
    else:
        savetxt(filename, dataset.x, delimiter=sep, fmt='%f')






if __name__ == '__main__':
    # test the function read_data_file
    #data = read_data_file('/Users/josediogomoura/machine_learning/datasets/datasets/iris.csv', sep=',', label=True)
    #print(data)
    data = read_data_file('/Users/josediogomoura/machine_learning/datasets/datasets/breast-bin.data', sep=',', label=False)
    # test the function write_data_file
    write_data_file(data, '/Users/josediogomoura/machine_learning/datasets/datasets/breast-bin.txt', sep=',', label=False)

