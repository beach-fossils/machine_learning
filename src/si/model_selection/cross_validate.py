from typing import Dict, List, Union, Callable

import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split

Num = Union[int, float]


def cross_validate(model, dataset: Dataset, scoring: Callable = None,
                   cv: int = 3, test_size: float = 0.3) -> Dict[str, List[Num]]:
    scores = {'seed': [], 'train': [], 'test': [], 'parameters': []}

    # calculates the score for each fold
    for seed in range(cv):

        # random seed
        random_state = np.random.randint(0, 1000)

        # store the seed
        scores['seed'].append(seed)

        # split the dataset
        train, test = train_test_split(dataset=dataset, test_size=test_size, random_state=random_state)

        # train the model
        model.fit(train)

        # score the model on the test set
        if scoring is None:

            # store the train score
            scores['train'].append(model.score(train))

            # store the test score
            scores['test'].append(model.score(test))

        else:
            y_train = train.y
            y_test = test.y

            # store the train score
            scores['train'].append(scoring(y_train, model.predict(train)))

            # store the test score
            scores['test'].append(scoring(y_test, model.predict(test)))

    return scores
