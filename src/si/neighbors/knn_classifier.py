from typing import Union, Callable
import numpy as np
from si.data.dataset import Dataset


class KNNClassifier:
    def __init__(self, k, distance):
        self.k = k  # number of neighbors to consider (numero de k exemplos)
        self.distance = distance  # distance function to be used (calcula a distancia entre amostra e as amostras do
        # dataset de treino)
        self.dataset = None  # armazena o dataset de treino

    def fit(self, dataset):
        """Stores the train dataset."""
        self.dataset = dataset

    def predict(self, dataset: Dataset) -> np.ndarray:
        """Estimates the class of each sample based on the k nearest neighbors and returns an array of estimated classes
        to the test dataset"""
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)  # np.apply_along_axis (aplica a
        # função _get_closest_label ao longo do eixo 1 do array dataset.X (dataset de teste))

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """Returns the label of the closest neighbor to the sample."""
        # compute the distance between the sample and all the samples in dataset
        distances = self.distance(sample, self.dataset.X)

        # get the indexes of the k smallest distances
        indexes = np.argsort(distances)[:self.k]

        # using indexes, get the labels of the k closest neighbors
        labels = self.dataset.y[indexes]

        # get the most frequent label
        label, _ = np.unique(labels,
                             return_counts=True)  # retorna os valores únicos (label) e a quantidade de vezes que
        # cada valor aparece (_) (return_counts=True)
        return label[np.argmax(_)]  # argmax returns the index of the maximum value in the array

    def score(self, dataset):
        """Computes the accuracy between the real and predicted values."""
        y_pred = self.predict(dataset) # y_pred é o array de classes estimadas
        accuracy = dataset.y == y_pred  # accuracy é um array de 1's e 0's, onde 1 indica que a classe estimada é igual
        # à classe real e 0 indica que a classe estimada é diferente da classe real
        return np.sum(accuracy) / len(accuracy)  # soma os elementos do array accuracy que são 1 e divide pelo número


# testing
if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

# a ser testado ainda!!!

