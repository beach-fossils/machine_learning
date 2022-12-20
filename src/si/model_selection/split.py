import numpy as np
from si.data.dataset import Dataset
from typing import Tuple


def train_test_split(dataset, test_size: float = 0.2, random_state=None) -> Tuple:
    """Splits the dataset into train and test sets, returning a tuple with train and test sets."""

    # set the seed for the random number generator
    if random_state is not None:
        np.random.seed(random_state)

    # get dataset size
    n_samples = len(dataset)  # n_samples é o número de amostras do dataset

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

    # create train and test sets
    train_set = Dataset(dataset.X[train_indexes], dataset.y[train_indexes],
                        features=dataset.features,
                        label=dataset.label)  # train_set é o dataset de treino, com as amostras de train_indexes
    test_set = Dataset(dataset.X[test_indexes], dataset.y[test_indexes],
                       features=dataset.features, label=dataset.label)  # test_set é o dataset de teste, features é
    # o nome das features e label é o nome da label

    return train_set, test_set
