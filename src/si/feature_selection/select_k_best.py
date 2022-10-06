import numpy as np
from si.data.dataset import Dataset
from src.si.statistics import f_classification


# score_func – função de análise da variância (f_classification) -> usar no teste

class SelectKBest:
    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset):
        """Estimate the F and p values for each feature"""
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset):
        """Returns the k best features."""
        idxs = np.argsort(self.F)[-self.k:]  # como é crescente, vamos buscar ao contrário (-)
        features = np.array(dataset.features)[idxs]
        return Dataset(x=dataset.x[:, idxs], y=dataset.y, features=features, label=dataset.label)  # retorna um novo
        # dataset com as features selecionadas

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)


# testing the methods
if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = ['a', 'b', 'c']
    label = 'y'
    dataset = Dataset(x, y, features, label)
    # print(dataset.summary())

    # testing fit method
    skb = SelectKBest(f_classification, k=1)
    print(skb.fit(dataset).F)

# avaliaçao: selcionar por maior F -> maior diferença nos dados
