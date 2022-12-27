import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset


class StackingClassifier:
    def __init__(self, models: list, final_model):
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset):
        """ Trains all models"""

        for model in self.models:
            model.fit(dataset)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """ Estimates the y using the trained models and final modelel"""

        # gets the model predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # gets the final model previsions
        y_pred = self.final_model.predict(Dataset(dataset.X, np.array(predictions).T))

        return y_pred

    def score(self, dataset) -> float:
        """ Returns the accuracy of the ensemble model"""

        y_pred = self.predict(dataset)
        score = accuracy(dataset.y, y_pred)
        return score
