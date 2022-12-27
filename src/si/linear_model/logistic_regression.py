from si.data import dataset
from src.si.statistics.sigmoid_function import sigmoid_function
import numpy as np
from src.si.data.dataset import Dataset
from matplotlib import pyplot as plt


class LogisticRegression:
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        self.l2_penalty = l2_penalty  # coeficiente de regularização
        self.alpha = alpha  # taxa de aprendizagem
        self.max_iter = max_iter  # número máximo de iterações

        self.theta = None  # parâmetro theta
        self.theta_zero = None  # parâmetro theta_zero
        self.cost_history = None

    def fit(self, dataset: Dataset):
        """Estimates the the theta and theta_zero parameters of the model using the dataset."""
        # similar to ridge_regression but uses sigmoid function

        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # initialize the cost history
        self.cost_history = {}

        # gradient descent, colocar para que o valor da função de custo (J/self.cost) não se altera
        for i in range(self.max_iter):
            # calculates the predictions using the sigmoid function
            y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

            # calculates the gradient
            gradient = (1 / m) * np.dot(dataset.X.T, y_pred - dataset.y)
            gradient_zero = (1 / m) * np.sum(y_pred - dataset.y)

            # calculates the regularization term
            regularization = (self.l2_penalty / m) * self.theta

            # updates the parameters
            self.theta = self.theta - self.alpha * (gradient + regularization)
            self.theta_zero = self.theta_zero - self.alpha * gradient_zero

            # calculates the cost function
            cost = self.cost_function(dataset)

            # updates the cost history
            self.cost_history[i] = cost

            # prints the cost function at every 20 iterations
            # if i % 100 == 0:
            #     print("Iteration: {}, Cost: {}".format(i, cost))

            # cost_history(i - 1) – cost_history(i) < 0.0001 : parar o algoritmo

            # prints the first and last iterations
            # if i == 0 or i == self.max_iter - 1:
            #     print("Iteration: {}, Cost: {}".format(i, cost))

            if i > 0:
                if self.cost_history[i - 1] - self.cost_history[i] < 0.0001:
                    break

        # plots the cost function history
        # self.cost_function_plot()

        # for i in range(self.max_iter):
        #     # predicted y
        #     y_pred = np.dot(dataset.X, self.theta) + self.theta_zero
        #
        #     # apply sigmoid function
        #     y_pred = sigmoid_function(y_pred)
        #
        #     # compute the gradient using the learning rate
        #     gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)
        #
        #     # compute the penalty
        #     penalization_term = self.alpha * (self.l2_penalty / m) * self.theta
        #
        #     # update the model parameters
        #     self.theta = self.theta - gradient - penalization_term
        #     self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)
        #
        #     # compute the cost function
        #     self.cost_history[i] = self.cost_function(dataset)

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

    def cost_function(self, dataset):
        """Calculates the cost function between the predicted values and the actual values."""
        # calculates the cost function

        m, n = dataset.shape()  # m = number of samples, n = number of features

        # estimates the y values using the sigmoid function
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # calculates the cost function using the predictions and the actual values
        cost = (1 / m) * np.sum(-dataset.y * np.log(y_pred) - (1 - dataset.y) * np.log(1 - y_pred))  # cost function
        cost = cost + (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)  # regularization term
        return cost

    def cost_function_plot(self):
        """Plots the cost function history."""
        plt.plot(self.cost_history.keys(), self.cost_history.values())
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
