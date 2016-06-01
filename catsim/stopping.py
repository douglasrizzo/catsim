from abc import ABCMeta, abstractmethod
from catsim import irt
import numpy


class Stopper(metaclass=ABCMeta):
    """Base class for CAT stop criterion"""

    def __init__(self):
        super(Stopper, self).__init__()

    @abstractmethod
    def stop(self, administered_items: numpy.ndarray, estimations: list) -> bool:
        """Checks whether the test reached its stopping criterion

        :param administered_itens: a matrix with the parameters of administered Items
        :param estimations: the estimations of the examinee's proficiency so far
        :returns: `True` if the test met its stopping criterion, else `False`"""
        pass


class MaxItemStopper(Stopper):
    """Stopping criterion for maximum number of items in a test

    :param max_itens: the maximum number of items in the test"""

    def __init__(self, max_itens: int):
        super(MaxItemStopper, self).__init__()
        self._max_itens = max_itens

    def stop(self, administered_items: numpy.ndarray, estimations: list) -> bool:
        """Checks whether the test reached its stopping criterion

        :param administered_itens: a matrix with the parameters of administered Items
        :param estimations: the estimations of the examinee's proficiency so far
        :returns: `True` if the test met its stopping criterion, else `False`"""
        n_itens = administered_items.shape[0]
        if n_itens > self._max_itens:
            raise ValueError(
                'More items than permitted were administered: {0} > {1}'.format(
                    n_itens, self._max_itens
                )
            )
        return n_itens == self._max_itens


class MinErrorStopper(Stopper):
    """Stopping criterion for minimum standard error of estimation (see :py:func:`catsim.irt.see`)

    :param min_error: the minimum standard error of estimation the test must achieve before stopping"""

    def __init__(self, min_error: float):
        super(MinErrorStopper, self).__init__()
        self._min_error = min_error

    def stop(self, administered_items: numpy.ndarray, estimations: list) -> bool:
        """Checks whether the test reached its stopping criterion

        :param administered_itens: a matrix with the parameters of administered Items
        :param estimations: the estimations of the examinee's proficiency so far
        :returns: `True` if the test met its stopping criterion, else `False`"""
        if len(estimations) == 0:
            return False

        theta = estimations[-1]
        return irt.see(theta, administered_items) < self._min_error
