import numpy as np


def accuracy(y_true, y_pred):
    """Computes the accuracy of the model, returning the error rate.
    Args:
        y_true: real values of y
        y_pred: estimated values of y
    Returns:
        The error rate.
    """
    return np.sum(y_true == y_pred) / len(y_true)  # um novo array é gerado (equal sign, true==pred) em que 1 indica
    # valores de true e pred são iguais e 0 indica valores de true e pred são diferentes e soma os elementos do array
    # que são 1 e divide pelo número de elementos do array



