import numpy as np


def train_test_split(dataset, test_size=0.2, random_state=None) -> tuple:
    """Splits the dataset into train and test sets and returns a tuple with train and test sets."""
    # generates permutation using np.random.permutation
    permutation = np.random.permutation(dataset.shape[0])

    # infer the number of samples in the test and train set
    test_size = int(dataset.shape[0] * test_size)
    train_size = int(dataset.shape[0] - test_size)

    # selects train and test using the permutation
    train = dataset.iloc[permutation[:train_size]]
    test = dataset.iloc[permutation[train_size:]]
    return train, test



