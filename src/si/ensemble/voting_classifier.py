import numpy as np
from si.metrics.accuracy import accuracy


class VotingClassifier:

    def __init__(self, models: list):
        self.models = models

    def fit(self, dataset):
        """ Trains all models"""

        for model in self.models:
            model.fit(dataset)

        return self

    def predict(self, X):
        """ Predicts the class of X based on the majority vote of the models"""

        # combina as previsoes de todos os modelos usando o voto maioritario
        predictions = np.array([model.predict(X) for model in self.models])  # (n_models, n_samples)

        def majority_vote(predictions: np.ndarray):
            # retorna o valor mais frequente
            labels, counts = np.unique(predictions, return_counts=True)
            return labels[np.argmax(counts)]

        # aplica a funcao de voto maioritario em cada coluna
        return np.apply_along_axis(majority_vote, axis=0, arr=predictions)

    def score(self, dataset):
        """ Returns the accuracy of the ensemble model"""

        y_pred = self.predict(dataset)
        score = accuracy(dataset.y, y_pred)
        return score
