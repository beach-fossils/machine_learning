from typing import Dict, Tuple, Callable, Union, List

import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate

Num = Union[int, float]


def randomize_search(model, dataset: Dataset, parameter_distribution: Dict[str, Tuple], scoring: Callable = None,
                     cv: int = 3, n_iter: int = 10, test_size: float = 0.3):
    scores = {
        'parameters': [],
        'seed': [],
        'train': [],
        'test': []
    }
    # Verifica se os parâmetros fornecidos existem no modelo
    for parameter in parameter_distribution.keys():
        if not hasattr(model, parameter):
            raise AttributeError(f"Missing parameter {parameter} in model {model.__class__.__name__}")

    # Obtem n_iter combinações aleatórias de parâmetros. Ou seja, se n_iter for igual a 10 deves obter 10
    # combinações dos parâmetros fornecidos. usar o random.choice para retirar um valor aleatório da distribuição de
    # valores de cada parâmetro
    parameter_combinations = []

    for i in range(n_iter):
        random_state = np.random.randint(0, 1000)

        scores['seed'].append(random_state)

        combination = {}
        for parameter, distribution in parameter_distribution.items():
            combination[parameter] = np.random.choice(distribution)

        # set the parameters to the model
        for parameter, value in combination.items():
            setattr(model, parameter, value)

        # cross validate the model
        cv_scores = cross_validate(model, dataset, scoring, cv, test_size)

        # save the parameters and scores
        for parameter, value in combination.items():
            scores['parameters'].append({parameter: value})
        for key in ['train', 'test']:
            scores[key].append(cv_scores[key])

    return scores
















