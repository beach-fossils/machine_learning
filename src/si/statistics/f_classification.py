from scipy import stats

from src import *


def f_classification(dataset):
    """  """

    classes = dataset.get_classes()
    groups = [dataset.x[dataset.y == c] for c in classes]
    f, p = stats.f_oneway(*groups)
    return f, p


def f_regression(dataset):

    pass
