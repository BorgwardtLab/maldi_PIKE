"""Base class for a dataset."""
import abc
from collections.abc import Sequence
from typing import List, Tuple

from maldi_learn.data import MaldiTofSpectrum


class Dataset(metaclass=abc.ABCMeta):
    """Abstract base class for a dataset."""

    @property
    @abc.abstractmethod
    def training_data(self) -> Tuple[List[MaldiTofSpectrum], Sequence]:
        """Get training data of dataset.

        Returns:
            Tuple (X, y)

        """

    @property
    @abc.abstractmethod
    def validation_data(self) -> Tuple[List[MaldiTofSpectrum], Sequence]:
        """Get validation data of dataset.

        Returns:
            Tuple (X, y)

        """

    @property
    @abc.abstractmethod
    def testing_data(self) -> Tuple[List[MaldiTofSpectrum], Sequence]:
        """Get testing data of dataset.

        Returns:
            Tuple (X, y)

        """

    @property
    @abc.abstractmethod
    def complete_data(self) -> Tuple[List[MaldiTofSpectrum], Sequence]:
        """Get complete dataset.

        Returns:
            Tuple (X, y)

        """
