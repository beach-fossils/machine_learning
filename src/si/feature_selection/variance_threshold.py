import numpy as np
import pandas as pd

from si.data import dataset
from si.data.dataset import Dataset


class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variance = None  # parametro estimado

    def fit(self, dataset: Dataset):
        """Estimate the variance for each feature"""
        # recebe o X do Dataset
        self.variance = np.var(dataset.X, axis=0)
        return self

    def tranform(self, dataset: Dataset) -> Dataset:
        """ It removes all features whose variance does not meet the threshold."""

        # retorna um vetor booleano com True para as features que passam no threshold
        mask = self.variance > self.threshold

        X = dataset.X[:, mask]  # retorna um novo Dataset com as features que passaram no threshold

        features = np.array(dataset.features)[mask]  # retorna um vetor com os nomes das features que passaram no
        # threshold

        return Dataset(X, dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset):
        """Estimate the variance for each feature and remove all features whose variance does not meet the threshold."""
        self.fit(dataset)
        return self.tranform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = ['a', 'b', 'c']
    label = 'y'
    dataset = Dataset(x, y, features, label)
    print(dataset.summary())

    # testing fit method
    vt = VarianceThreshold()
    print(vt.fit_transform(dataset))
