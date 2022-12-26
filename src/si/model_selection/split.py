import numpy as np
from si.data.dataset import Dataset
from typing import Tuple


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state=None) -> Tuple:
    """Splits the dataset into train and test sets, returning a tuple with train and test sets."""

    # seed for the random number generator
    np.random.seed(random_state)

    # size of the test set
    n_samples = dataset.shape()[0]  # number of samples is the number of rows of the dataset
    split_div = int(n_samples * test_size)  # number of samples in the test set is the number of samples times the test
    # size

    # get the dataset permutations
    permutations = np.random.permutation(n_samples)  # get a permutation of the number of samples

    # get the test and train sets
    test_idx = permutations[:split_div]  # get the first split_div samples of the permutation
    train_idx = permutations[split_div:]  # get the remaining samples of the permutation

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idx], dataset.y[train_idx], features=dataset.features,
                    label=dataset.label)

    test = Dataset(dataset.X[test_idx], dataset.y[test_idx], features=dataset.features,
                   label=dataset.label)

    return train, test


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
