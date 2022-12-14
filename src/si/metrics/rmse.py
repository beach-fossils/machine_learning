import numpy as np


def rmse(y_true, y_pred) -> error:
    """Computes the root mean squared error between the real and predicted values.
    Args:
        y_true: real values of y
        y_pred: estimated values of y
    Returns:
        The root mean squared error.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))  # raiz quadrada da média dos quadrados das diferenças entre os
    # valores reais e os valores estimados
