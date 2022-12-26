import numpy as np
from src.si.metrics.rmse import rmse
from src.si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:

    def __init__(self, k, distance=euclidean_distance):
        self.k = k  # numero de vizinhos a considerar
        self.distance = distance  # função de distancia a ser usada
        self.dataset = None

    def fit(self, dataset):
        self.dataset = dataset

    def predict(self, dataset):
        """ Similar to KNNClassifier, but instead of returning the most frequent label, it returns the mean of the k
        closest neighbors"""
        return np.apply_along_axis(self._get_mean, axis=1, arr=dataset.X)

    def score(self, dataset):
        """The method obtains predictions and calculates the rsme between the real and predicted values."""
        return rmse(dataset.y, self.predict(dataset))

    def _get_mean(self, sample):
        """Returns the mean of the k closest neighbors."""
        distances = self.distance(sample, self.dataset.X)  # calcula a distancia entre a amostra e as amostras do
        # dataset de treino
        indexes = np.argsort(distances)[:self.k]  # retorna os indices dos k menores valores de distances, o argsort
        # retorna
        # os indices dos valores ordenados de distances
        labels = self.dataset.y[indexes]  # retorna as classes dos k vizinhos mais próximos
        return np.mean(labels)  # retorna a média das classes dos k vizinhos mais próximos


# testing
if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.io.CSV import read_csv

    # # load the dataset
    # iris_dataset = read_csv('/Users/josediogomoura/machine_learning/datasets/datasets/iris.csv', sep=',', features=True,
    #                         label=True)
    #
    # # split the dataset into train and test
    # train, test = train_test_split(iris_dataset, test_size=0.2)
    train = Dataset(X=np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]),
                    y=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    test = Dataset(X=np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]),
                     y=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    # # create the model
    knn = KNNRegressor(k=3)

    # fit the model
    knn.fit(train)

    # predict
    predictions = knn.predict(test)

    # score
    print(knn.score(test))

    # accuracy
    print(accuracy(test.y, predictions))


