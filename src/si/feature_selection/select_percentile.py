import numpy as np

from si.data.dataset import Dataset


class SelectPercentile:
    def __init__(self, score_func, percentile):
        self.score_func = score_func # função de análise da variância (f_classification)
        self.percentile = percentile # percentil de features a selecionar
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset):
        """Estimate the F and p values for each feature"""
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset):
        """Returns the k best features."""
        idxs = np.argsort(self.F)[-self.k:] # argsort -> ordena os valores e retorna os indices
        features = np.array(dataset.features)[idxs]
        return Dataset(x=dataset.x[:, idxs], y=dataset.y, features=features, label=dataset.label) # retorna um novo

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)


