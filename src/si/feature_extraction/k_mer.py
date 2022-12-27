import itertools
from typing import List

import numpy as np
from numpy import ndarray

from si.data.dataset import Dataset


# get the sequences in the dataset, then get the k-mers using itertools and the sequences and the k-mer length


class KMer:
    def __init__(self, k=3, alphabet='DNA'):

        self.k = k  # tamanho da substring
        self.alphabet = alphabet.upper()  # alfabeto pode ser DNA ou peptidos

        self.k_mers = None
        self.sequences = None

        if self.alphabet == 'DNA':
            self.alphabet = 'ACGT'
        elif self.alphabet == 'PEPTIDE':
            self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'

    def fit(self, dataset: Dataset):
        """Fits the k-mer model.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model.
        """
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]

        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """Transforms the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to transform.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        sequencer = [self._get_k_mers(sequence) for sequence in dataset.X[:, 0]]

        return Dataset(sequencer, dataset.y, label=dataset.label, features=self.k_mers)

    def _get_k_mers(self, sequence: str) -> ndarray:
        """Gets the k-mers from the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to get the k-mers.

        Returns
        -------
        ndarray
            The k-mers.
        """
        k_mers = np.zeros(len(self.k_mers))

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            k_mers[self.k_mers.index(k_mer)] += 1

        return k_mers

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Fits and transforms the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit and transform.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)
