import numpy as np
import pandas as pd

from si.data import dataset


class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variance = None

    def fit(self, x):
        self.variance = np.var(x, axis=0)
        return self

    def tranform(self, x):
        mask = self.variance > self.threshold  # mask is a boolean array
        x = x[:, mask]
        features = np.array(dataset.features)[mask]
        return x, features

    def fit_transform(self, x):
        self.fit(x)
        return self.tranform(x)


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
    print(vt.fit_transform(x))
