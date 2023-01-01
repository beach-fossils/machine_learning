import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Cross entropy loss function

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    np.ndarray
        Cross entropy loss.
    """
    return -np.sum(y_true * np.log(y_pred), axis=1)


def cross_entropy_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Cross entropy loss derivative

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    np.ndarray
        Cross entropy loss derivative.
    """
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)


