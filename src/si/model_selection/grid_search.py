from itertools import product
from typing import Callable, Dict, Tuple, List
from itertools import product
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def grid_search_cv(model, dataset: Dataset, parameter_grid: Dict, scoring: Callable = None,
                   cv: int = 3, test_size: float = 0.3) -> Dict[str, List[float]]:
    # Verifica se os parâmetros fornecidos existem no modelo. use the hasattr function
    # to check if the model has the attribute
    for parameter in parameter_grid.keys():
        if not hasattr(model, parameter):
            raise AttributeError(f"Missing parameter {parameter} in model {model.__class__.__name__}")

    # Obtém o produto cartesiano dos parâmetros fornecidos (todas as combinações possíveis). Podes usar
    # o itertools.product para obter todas as combinações.
    parameter_combinations = list(product(*parameter_grid.values()))

    # seguir os proximos passos:
    # 1. Altera os parâmetros do modelo com uma combinação. Podes usar a função do python setattr.
    # 2. Realiza o cross_validate com esta combinação
    # 3. Guarda a combinação de parâmetros e os scores obtidos.
    # 4. Repete os passos 1 a 3 para todas as combinações de parâmetros.
    # O grid_search deve retornar uma lista de dicionários. Os dicionários devem conter as
    # seguintes chaves: 'parameters', 'train', 'test', 'seed'
    #

    scores = {'parameters': [], 'train': [], 'test': [], 'seed': []}
    for combination in parameter_combinations:
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
        cv_scores = cross_validate(model, dataset, scoring, cv, test_size)
        for parameter, value in zip(parameter_grid.keys(), combination):
            scores['parameters'].append({parameter: value})
        for key in ['train', 'test', 'seed']:
            scores[key] += cv_scores[key]

    return scores



















