import numpy as np
from src.si.statistics.sigmoid_function import sigmoid_function


class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.
    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        return error


class SigmoidActivation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        return sigmoid_function(X)

    def backward(self, X: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Performs a backward pass of the layer using the given error.
        Returns a 2d numpy array with shape (1, input_size).
        Parameters
        ----------
        error: np.ndarray
            The error of the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """

        derivate = sigmoid_function(X) * (1 - sigmoid_function(X))
        return error * derivate


class SoftMaxActivation:
    """Esta layer deve calcular a probabilidade de ocorrência de cada classe. Útil para problemas multiclass.
    """

    def __init__(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """

        ezi = np.exp(X - np.max(X))
        formula = ezi / np.sum(ezi, axis=1, keepdims=True)

        return formula

    def backward(self, X: np.ndarray, error: np.ndarray) -> np.ndarray:
        derivate = sigmoid_function(X) * (1 - sigmoid_function(X))
        return error * derivate


class ReLUActivation:
    def __init__(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    def backward(self, X: np.ndarray, error: np.ndarray) -> np.ndarray:
        derivate = np.where(X > 0, 1, 0)
        return error * derivate


class LinearActivation:
    def __init__(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X
