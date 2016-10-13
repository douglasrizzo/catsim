import numpy

from catsim import irt
from catsim.simulation import Stopper


class MaxItemStopper(Stopper):
    """Stopping criterion for maximum number of items in a test

    :param max_itens: the maximum number of items in the test"""

    def __str__(self):
        return 'Maximum Item Number Initializer'

    def __init__(self, max_itens: int):
        super(MaxItemStopper, self).__init__()
        self._max_itens = max_itens

    def stop(self, index: int = None, administered_items: numpy.ndarray = None, **kwargs) -> bool:
        """Checks whether the test reached its stopping criterion for the given user

        :param index: the index of the current examinee
        :param administered_items: a matrix containing the parameters of items that were already administered
        :returns: `True` if the test met its stopping criterion, else `False`"""

        if (index is None or self.simulator is None) and administered_items is None:
            raise ValueError

        if administered_items is None:
            administered_items = self.simulator.items[self.simulator.administered_items[index]]

        n_itens = administered_items.shape[0]
        if n_itens > self._max_itens:
            raise ValueError('More items than permitted were administered: {0} > {1}'.format(n_itens, self._max_itens))

        return n_itens == self._max_itens


class MinErrorStopper(Stopper):
    """Stopping criterion for minimum standard error of estimation (see :py:func:`catsim.irt.see`)

    :param min_error: the minimum standard error of estimation the test must achieve before stopping"""

    def __str__(self):
        return 'Minimum Error Initializer'

    def __init__(self, min_error: float):
        super(MinErrorStopper, self).__init__()
        self._min_error = min_error

    def stop(self, index: int = None, administered_items: numpy.ndarray = None, theta: float = None, **kwargs) -> bool:
        """Checks whether the test reached its stopping criterion

        :param index: the index of the current examinee
        :param administered_items: a matrix containing the parameters of items that were already administered
        :param theta: a float containing the a proficiency value to which the error will be calculated
        :returns: `True` if the test met its stopping criterion, else `False`"""

        if (index is None or self.simulator is None) and (administered_items is None or theta is None):
            raise ValueError

        if administered_items is None and theta is None:
            theta = self.simulator.latest_estimations[index]
            administered_items = self.simulator.items[self.simulator.administered_items[index]]

        if theta is None:
            return False

        return irt.see(theta, administered_items) < self._min_error
