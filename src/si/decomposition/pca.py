import numpy as np
import pandas as pd

#  Técnica de álgebra linear para reduzir as dimensões do dataset. O PCA a implementar usa a técnica de álgebra linear
#  SVD (Singular Value Decomposition)

from si.data.dataset import Dataset


class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset):
        self.components = self._get_components(dataset)
        self.explained_variance = self._get_explained_variance(dataset)

        return self

    def _get_centered_data(self, dataset: Dataset) -> np.ndarray:
        """
        Centers the dataset.
        :param dataset: Dataset object.
        :return: A matrix with the centered data.
        """

        self.mean = np.mean(dataset.X, axis=0)  # axis=0 means that we want to calculate the mean for each column
        return dataset.X - self.mean

    def _get_explained_variance(self, dataset: Dataset) -> np.ndarray:
        """
        Calculates the explained variance.
        :param dataset: Dataset object.
        :return: A vector with the explained variance.
        """

        ev_formula = np.var(dataset.X, axis=0) / np.sum(np.var(dataset.X, axis=0))  # axis=0 means that we want to calculate
        # the variance for each column and then we divide by the sum of all variances
        # a variante explicada corresponde aos primeiros n_componentes
        explained_var = ev_formula[:self.n_components]

        return explained_var

    def _get_components(self, dataset: Dataset) -> np.ndarray:
        """
        Calculates the components.
        :param dataset: Dataset object.
        :return: A matrix with the components.
        """
        # centrar os dados
        centered_data = self._get_centered_data(dataset)

        # decomposição SVD
        self.u, self.s, self.vh = np.linalg.svd(centered_data)  # u é a matriz U, s é o vetor S e vh é a matriz V transposta,
        # U é uma matriz com as mesmas dimensões que os dados centrados, S é um vetor com o número de colunas dos dados
        # e o V transposta é uma matriz com o número de linhas dos dados centrados ~???

        # obter os componentes principais; .T
        components = self.vh[:self.n_components].T  # através do vh obtemos os componentes principais

        return components

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset.
        :param dataset: Dataset object.
        :return: A matrix with the transformed data.
        """

        # centrar os dados
        centered_data = self._get_centered_data(dataset)  # centered_data = dataset.X - self.mean

        # transposta de V
        v_t = self.vh.T

        # multiplicar os dados centrados pela transposta de V
        transformed_data = np.dot(centered_data, v_t)

        return Dataset(transformed_data, dataset.y, dataset.features, dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits and transforms the dataset.
        :param dataset: Dataset object.
        :return: A matrix with the transformed data.
        """

        self.fit(dataset)
        return self.transform(dataset)



