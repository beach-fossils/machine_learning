# X = numpy array of shape (n_samples, n_features)
# Y = numpy array of shape (n_samples, 1) uma dimensao
# features = lista de strings com os nomes das features
# label = string com o nome da label

import numpy as np
import pandas as pd


class Dataset:
    """A class to represent a dataset.
    """

    def __init__(self, X, y=None, features=None, label=None):
        """Initializes the dataset.
        Args:
            X: A numpy array of shape (n_samples, n_features).
            y: A numpy array of shape (n_samples, 1).
            features: A list of strings with the names of the features.
            label: A string with the name of the label.
            """

        self.X = X  # variáveis independentes, matriz de features
        self.y = y  # variável dependente
        self.features = features  # vetor com nomes das features
        self.label = label  # nome da label

    def shape(self):
        """Returns the shape of the dataset."""
        return self.X.shape

    def has_labels(self):
        """Returns True if the dataset has labels.
        if true, the dataset is a supervised dataset."""
        return self.y is not None  # se y for None, retorna False

    def get_classes(self):
        """Possible values in y; returns the classes of the dataset."""
        if self.y is None:
            raise ValueError('Dataset has no labels')
        return np.unique(self.y)

    def get_mean(self):
        """Returns the mean of the dataset."""
        return np.mean(self.X, axis=0)

    def get_variance(self):
        """Returns the variance of the dataset."""
        return np.var(self.X, axis=0)

    def get_median(self):
        """Returns the median of the dataset."""
        return np.median(self.X, axis=0)

    def get_min(self):
        """Returns the minimum of the dataset."""
        return np.min(self.X, axis=0)

    def get_max(self):
        """Returns the maximum of the dataset."""
        return np.max(self.X, axis=0)

    def summary(self):
        """Returns a summary of the dataset."""
        # pandas dataframe colum names are features and row mean, median, min, max, variance
        df = pd.DataFrame(columns=self.features)
        df.loc['mean'] = self.get_mean()
        df.loc['median'] = self.get_median()
        df.loc['min'] = self.get_min()
        df.loc['max'] = self.get_max()
        df.loc['variance'] = self.get_variance()
        return df

    def dropna(self):
        """Removes all the samples with missing values (NaN)."""

        mask = np.isnan(self.X).any(axis=1)  # retorna um vetor de booleanos com True se houver NaN na linha
        self.X = self.X[~mask]  # inverte a mascara para manter os valores que nao sao NaN e assim ficamos com os
        # valores que nao sao NaN

    def fillna(self, value):
        """Fills all the missing values (NaN) with a given value."""

        self.X = np.nan_to_num(self.X, nan=value)

        # update y if it exists
        if self.has_labels():
            self.y = np.nan_to_num(self.y, nan=value)


# testing

if __name__ == '__main__':
    pass
