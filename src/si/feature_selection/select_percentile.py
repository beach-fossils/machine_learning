import numpy as np

from si.data.dataset import Dataset


class SelectPercentile:

    def __init__(self, score_func, percentile):
        self.score_func = score_func  # função de análise da variância (f_classification)
        self.percentile = percentile  # percentil de features a selecionar
        self.F = None
        self.p = None

    def fit(self, dataset):
        """Estimate the F and p values for each feature"""
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        """Selects the k features compatible with the percentile desired to be applied on F values and returns the
        dataset with the features that passed the percentile threshold. """

        # saber quais são as features que passam no threshold do percentil atrvés dos valores de F e ids
        idxs = np.argsort(self.F)  # retorna os indices das features em ordem crescente de F
        n_features = int(len(idxs) * self.percentile)  # número de features a selecionar
        idxs = idxs[-n_features:]  # seleciona as features com os maiores valores de F de acordo com o percentil, o - é
        # para inverter a ordem pois o argsort retorna em ordem crescente

        features = np.array(dataset.features)[idxs]  # retorna um vetor com os nomes das features em ordem crescente de
        # F

        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset):
        """Estimate the F and p values for each feature and returns the k best features."""
        self.fit(dataset)
        return self.transform(dataset)


