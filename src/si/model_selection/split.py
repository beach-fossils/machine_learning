import numpy as np
from si.data.dataset import Dataset
from typing import Tuple


def train_test_split(dataset, test_size: float = 0.2, random_state=None) -> Tuple:
    """Splits the dataset into train and test sets, returning a tuple with train and test sets."""

    # set the seed for the random number generator
    if random_state is not None:
        np.random.seed(random_state)

    # get dataset size
    n_samples = dataset.shape()[0]  # n_samples é o número de amostras do dataset

    # get number of samples in the test set
    n_test_samples = int(n_samples * test_size)  # n_test_samples é o número de amostras do dataset de teste

    # get dataset permutation
    permutation = np.random.permutation(n_samples)  # permutation é um array com os índices das amostras do dataset

    # get indexes of the test set
    test_indexes = permutation[:n_test_samples]  # test_indexes é um array com os índices das amostras do dataset de
    # teste

    # get indexes of the train set
    train_indexes = permutation[n_test_samples:]  # train_indexes é um array com os índices das amostras do dataset de
    # treino

    # create train and test sets it is giving me error of NoneType in the next line
    train_set = Dataset(X=dataset.X[train_indexes], y=dataset.y[train_indexes], features=dataset.features,
                        label=dataset.label)
    test_set = Dataset(X=dataset.X[test_indexes], y=dataset.y[test_indexes], features=dataset.features,
                       label=dataset.label)

    return train_set, test_set


# testing the function
if __name__ == '__main__':
    from src.si.io.CSV import read_csv

    # load the dataset
    dataset = read_csv('/Users/josediogomoura/machine_learning/datasets/datasets/iris.csv', sep=',', features=True,
                       label=True)

    # split the dataset
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    # print the train set
    print(train_set.X)
    print(train_set.y)

    # print the test set
    print(test_set.X)
    print(test_set.y)
