from si.data import dataset
from src.si.statistics.sigmoid_function import sigmoid_function
import numpy as np
from src.si.data.dataset import Dataset


class LogisticRegression:
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        self.l2_penalty = l2_penalty  # coeficiente de regularização
        self.alpha = alpha  # taxa de aprendizagem
        self.max_iter = max_iter  # número máximo de iterações

        self.theta = None  # vetor de pesos, parametros do modelo para as variáveis de entrada
        # The intercept of the model.
        self.theta_zero = None  # coeficiente/parametro zero. Interserção

    def fit(self, dataset: Dataset):
        """Estimates the the theta and theta_zero parameters of the model using the dataset."""
        # similar to ridge_regression but uses sigmoid function

        m, n = dataset.shape()  # m = number of samples, n = number of features
        # gradient descent
        for i in range(self.max_iter):
            # predict y using sigmoid function
            y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

            # calculate gradients
            gradient_theta = (1 / m) * np.dot(dataset.X.T, (y_pred - dataset.y)) + (self.l2_penalty / m) * self.theta
            gradient_theta_zero = (1 / m) * np.sum(y_pred - dataset.y)  # gradient of theta_zero is not regularized

            # update parameters theta and theta_zero using the gradients
            self.theta = self.theta - self.alpha * gradient_theta
            self.theta_zero = self.theta_zero - self.alpha * gradient_theta_zero

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """Estimates the dependent variable for the dataset using the model."""
        # predict method  estimates the values of y using theta, theta_zero and function sigmoid
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # convert predictions to 0 or 1
        predictions[predictions >= 0.5] = 1  # if prediction >= 0.5, then y = 1
        predictions[predictions < 0.5] = 0  # if prediction < 0.5, then y = 0

        return predictions

    def score(self, dataset):
        """Calculates the error between the predicted values and the actual values."""
        # obtains the predictions using the predict method
        predictions = self.predict(dataset)

        # calculates accuracy between predictions and dataset.y
        return np.sum(predictions == dataset.y) / len(dataset.y)

    def cost(self, dataset):
        """Calculates the cost function between the predicted values and the actual values."""
        # calculates the cost function

        m, n = dataset.shape()  # m = number of samples, n = number of features

        # estimates the y values using the sigmoid function
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # calculates the cost function using the predictions and the actual values
        cost = (1 / m) * np.sum(-dataset.y * np.log(y_pred) - (1 - dataset.y) * np.log(1 - y_pred))  # cost function
        cost = cost + (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)  # regularization term
        return cost

