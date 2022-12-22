import numpy as np


def euclidean_distance(x, y):
    """Compute the Euclidean distance between two vectors.
    param X: numpy array 1D, one sample
    param y: numpy array variable dimension, many samples"""

    return np.sqrt((x - y) ** 2).sum(axis=1)  # distancia euclidiana é normalmente vetor a vetor, este vetor retornado
    # é a distancia entre x e cada um dos y




