import numpy as np
from matplotlib import pyplot as plt

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    cost_history: dict
        The cost function history for each iteration
    """

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        """
        # parameters
        self.l2_penalty = l2_penalty  # coeficiente de regularização
        self.alpha = alpha  # a learning rate
        self.max_iter = max_iter  # número máximo de iterações

        # attributes
        self.theta = None  # coeficientes/parametros do modelo para as variáveis de entrada (features)
        self.theta_zero = None  # coeficiente/parametro zero. também conhhecido como interseção
        self.cost_history = None

    def fit(self, dataset: Dataset):
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegression
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # cost history iniciar
        self.cost_history = {}

        # gradient descent
        for i in range(self.max_iter):
            y_pred = self.predict(dataset)

            # compute the gradients
            d_theta = (np.dot(dataset.X.T, (y_pred - dataset.y)) + (self.l2_penalty * self.theta)) / m
            d_theta_zero = np.sum(y_pred - dataset.y) / m

            # update the parameters
            self.theta = self.theta - (self.alpha * d_theta)
            self.theta_zero = self.theta_zero - (self.alpha * d_theta_zero)

            # compute the cost function
            self.cost_history[i] = self.cost(dataset)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Estimates the values of y for the dataset using theta and theta zero.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        y_pred = self.predict(dataset)

        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))

    def cost_function_plot(self):
        """
        Plot the cost function history
        """
        plt.plot(list(self.cost_history.keys()), list(self.cost_history.values()))
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegression()
    model.fit(dataset_)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score = model.score(dataset_)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f"Predictions: {y_pred_}")
