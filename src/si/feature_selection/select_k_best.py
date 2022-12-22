import numpy as np
from si.data.dataset import Dataset
from src.si.statistics import f_classification
from typing import Callable


# score_func – função de análise da variância (f_classification) -> usar no teste

class SelectKBest:
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset):
        """Estimate the F and p values for each feature"""
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """ Selects the k features with highest F values and returns the dataset with highest scoring features. """

        idxs = np.argsort(self.F)[-self.k:]  # retorna os indices das k features com maior F, como é um vetor ordenado,
        # usa-se o -k para ter os ultimos k elementos
        features = np.array(dataset.features)[idxs]  # retorna um vetor com os nomes das features que passaram no
        # threshold
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)



