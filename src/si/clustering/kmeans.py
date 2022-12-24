from collections.abc import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance


class KMeans:
    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        # parameters
        self.k = k  # number of clusters
        self.max_iter = max_iter  # maximum number of iterations
        self.distance = distance  # distance function

        # attributes
        self.centroids = None  # média das amostras em cada cluster
        self.labels = None  # vetor com label de cada cluster

    def fit(self, dataset):
        """It fits k-means clustering on the dataset.
        This algorithm initializes k number of centroids randomly and then iterates through the dataset to find the
        closest centroid for each sample. After that, it calculates the mean of each cluster and updates the centroids.
        Args:
            dataset (Dataset): Dataset object
        Returns:
            KMeans: KMeans object"""
        self._init_centroids(dataset)

        convergence = False
        iteration = 0
        labels = np.zeros(dataset.shape()[0])  # vetor com o label de cada cluster

        while not convergence and iteration < self.max_iter:
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1,
                                             arr=dataset.X)  # calcula a distancia entre
            # cada amostra e cada centroide e retorna o indice do centroide mais proximo
            centroids = []

            for i in range(self.k):
                cluster = dataset.X[
                    new_labels == i]  # as amostras que foram classificadas como pertencentes ao cluster i
                # são as amostras que tem o indice i no vetor new_labels
                centroids.append(np.mean(cluster, axis=0))  # calcula a media de cada cluster

            convergence = np.array_equal(labels, new_labels)  # verifica se os labels antigos são iguais aos novos
            labels = new_labels  # atualiza os labels
            self.centroids = np.array(centroids)  # atualiza os centroides com os novos valores

            iteration += 1

        self.labels = labels  # atualiza os labels finais

        return self

    def _init_centroids(self, dataset):
        """Creates randomly the centroids."""
        seeds = np.random.permutation(dataset.shape[0])[:self.k]  # seleciona atraves de permutação aleatória os indices
        # dos centroides iniciais (k) a partir do dataset (dataset.shape[0]), que é o número de linhas do dataset,
        # o [:self.k] é para usar apenas os k primeiros indices da permutação
        self.centroids = dataset.X[seeds]  # buscar amostras ao dataset X. exemplo se k = 2, então cada centroide tem 2
        # vetores

    def _get_closest_centroid(self, sample: np.array) -> np.array:
        """Calculates the shortest distance between the sample X and the centroids."""
        # uses np.apply_along_axis to calculate the distance between the sample X and each centroid
        centroids_distances = self.distance(sample, self.centroids)  # distances array. X to various y
        closest_centroid_index = np.argmin(centroids_distances, axis=0)  # index of the closest centroid. argmin dá o
        # index do valor menor naquele vetor
        return closest_centroid_index

    def transform(self, dataset: Dataset) -> np.ndarray:
        """Calculates the distance between each sample and each centroid."""
        # queremos as distancias entre cada amostra e cada centroide
        centroids_distances = np.apply_along_axis(self._get_distance,
                                                  axis=1, arr=dataset.X)
        return centroids_distances

    def _get_distance(self, sample: np.array, centroid: np.array) -> float:
        """Calculates the distance between the sample X and the centroid."""
        return self.distance(sample, centroid)

    def predict(self, x: Dataset) -> np.ndarray:
        """Infers which the centroids are the closest to the sample X."""
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=x.X)

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """Fits and transforms the dataset."""
        self.fit(dataset)
        return self.transform(dataset)

