"""Module containing functions relevant to the process of simulating the
application of adaptive tests. Most of this module is based on the work of
[Bar10]_."""

import datetime
import time
import tqdm
from abc import ABCMeta, abstractmethod

import numpy

from catsim import cat, irt


class Simulable(metaclass=ABCMeta):
    """Base class representing one of the Simulator components that will receive a reference back to it."""

    def __init__(self):
        super(Simulable).__init__()
        self._simulator = None

    @property
    def simulator(self):
        if self._simulator is not None and type(self._simulator) is not Simulator:
            raise ValueError('simulator has to be of type catsim.simulation.Simulator')
        return self._simulator

    @simulator.setter
    def simulator(self, x: 'Simulator'):
        if type(x) is not Simulator:
            raise ValueError('simulator has to be of type catsim.simulation.Simulator')
        self._simulator = x
        self.preprocess()

    def preprocess(self):
        """Override this method to initialize any static values the `Simulable` might use for the duration of the
        simulation. `preprocess` is called after a value is set for the `simulator` property. If a new value if
        attributed to `simulator`, this method is called again, guaranteeing that internal properties of the
        `Simulable` are re-initialized as necessary."""
        pass


class Initializer(Simulable, metaclass=ABCMeta):
    """Base class for CAT initializers"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def initialize(self, index: int) -> float:
        """Selects an examinee's initial :math:`\\theta` value

        :param index: the index of the current examinee
        :returns: examinee's initial :math:`\\theta` value
        """
        pass


class Selector(Simulable, metaclass=ABCMeta):
    """Base class representing a CAT item selector."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def select(self, index: int) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied or `None` if there are no more items to be presented.
        """
        pass


class FiniteSelector(Selector, metaclass=ABCMeta):
    """Base class representing a CAT item selector."""

    def __init__(self, test_size):
        self._test_size = test_size
        self._overlap_rate = None
        super().__init__()

    @property
    def test_size(self):
        return self._test_size

    @property
    def overlap_rate(self) -> float:
        return self._overlap_rate


class Estimator(Simulable, metaclass=ABCMeta):
    """Base class for proficiency estimators"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def estimate(self, index: int) -> float:
        """Returns the theta value that minimizes the negative log-likelihood function, given the current state of the
         test for the given examinee.

        :param index: index of the current examinee in the simulator
        :returns: the current :math:`\\hat\\theta`
        """
        pass


class Stopper(Simulable, metaclass=ABCMeta):
    """Base class for CAT stop criterion"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def stop(self, index: int) -> bool:
        """Checks whether the test reached its stopping criterion for the given user

        :param index: the index of the current examinee
        :returns: `True` if the test met its stopping criterion, else `False`"""
        pass


class Simulator:
    """Class representing the simulator. It gathers several objects that describe the full
    simulation process and simulates one or more computerized adaptive tests

    :param items: a matrix containing item parameters
    :param examinees: an integer with the number of examinees, whose real :math:`\\theta` values will be
                      sampled from a normal distribution; or a :py:type:list containing said
                      :math:`\\theta_0` values
    """

    def __init__(
        self,
        items: numpy.ndarray,
        examinees,
        initializer: Initializer = None,
        selector: Selector = None,
        estimator: Estimator = None,
        stopper: Stopper = None
    ):
        irt.validate_item_bank(items)

        # adds a column for each item's exposure rate
        if items.shape[1] < 5:
            items = numpy.append(items, numpy.zeros([items.shape[0], 1]), axis=1)

        self._duration = 0
        self._items = items

        self._bias = None
        self._mse = None
        self._rmse = None
        self._overlap_rate = None

        self._initializer = initializer
        self._selector = selector
        self._estimator = estimator
        self._stopper = stopper

        # `examinees` is passed to its special setter
        self.examinees = examinees

        self._estimations = [[] for _ in range(len(self.examinees))]
        self._administered_items = [[] for _ in range(len(self.examinees))]
        self._response_vectors = [[] for _ in range(len(self.examinees))]

    @property
    def items(self) -> numpy.ndarray:
        """Item matrix used by the simulator. If the simulation already
        occurred, a column containing item exposure rates will be added to the
        matrix."""
        return self._items

    @property
    def administered_items(self) -> list:
        """List of lists containing the indexes of items administered to each
        examinee during the simulation."""
        return self._administered_items

    @property
    def estimations(self) -> list:
        """List of lists containing all estimated :math:`\\hat\\theta` values
        for all examinees during each step of the test."""
        return self._estimations

    @property
    def response_vectors(self) -> list:
        """List of boolean lists containing the examinees answers to all items."""
        return self._response_vectors

    @property
    def latest_estimations(self) -> list:
        """Final estimated :math:`\\hat\\theta` values for all examinees."""
        return [ests[-1] if len(ests) > 0 else None for ests in self._estimations]

    @property
    def examinees(self) -> list:
        """List containing examinees true proficiency values (:math:`\\theta`)."""
        return self._examinees

    @property
    def duration(self) -> float:
        """Duration of the simulation, in seconds."""
        return self._duration

    @property
    def overlap_rate(self) -> float:
        """Overlap rate of the test, if it is of finite length."""
        return self._overlap_rate

    @property
    def initializer(self) -> Initializer:
        return self._initializer

    @property
    def selector(self) -> Selector:
        return self._selector

    @property
    def estimator(self) -> Estimator:
        return self._estimator

    @property
    def stopper(self) -> Stopper:
        return self._stopper

    @property
    def bias(self) -> float:
        """Bias between the estimated and true proficiencies. This property is only
        available after :py:func:`simulate` has been successfully called. For more
        information on estimation bias, see :py:func:`catsim.cat.bias`"""
        return self._bias

    @property
    def mse(self) -> float:
        """Mean-squared error between the estimated and true proficiencies. This
        property is only available after :py:func:`simulate` has been successfully
        called. For more information on the mean-squared error of estimation, see
        :py:func:`catsim.cat.mse`"""
        return self._mse

    @property
    def rmse(self) -> float:
        """Root mean-squared error between the estimated and true proficiencies. This
        property is only available after :py:func:`simulate` has been successfully
        called. For more information on the root mean-squared error of estimation, see
        :py:func:`catsim.cat.rmse`"""
        return self._rmse

    @examinees.setter
    def examinees(self, x):
        if type(x) == int:
            if self._items is not None:
                mean = numpy.mean(self._items[:, 1])
                stddev = numpy.std(self._items[:, 1])
                self._examinees = numpy.random.normal(mean, stddev, x)
            else:
                self._examinees = numpy.random.normal(0, 1, x)
        elif type(x) == list:
            self._examinees = numpy.array(x)
        elif type(x) == numpy.ndarray and x.ndim == 1:
            self._examinees = x
        else:
            raise ValueError(
                'Examinees must be an int, list of floats or one-dimensional numpy array'
            )

    def simulate(
        self,
        initializer: Initializer = None,
        selector: Selector = None,
        estimator: Estimator = None,
        stopper: Stopper = None,
        verbose: bool = False
    ):
        """Simulates a computerized adaptive testing application to one or more examinees

        :param initializer: an initializer that selects examinees :math:`\\theta_0`
        :param selector: a selector that selects new items to be presented to examinees
        :param estimator: an estimator that reestimates examinees proficiencies after each item is applied
        :param stopper: an object with a stopping criteria for the test
        :param verbose: whether to periodically print a message regarding the progress of the simulation.
                        Good for longer simulations.

        >>> from catsim.initialization import RandomInitializer
        >>> from catsim.selection import MaxInfoSelector
        >>> from catsim.estimation import HillClimbingEstimator
        >>> from catsim.stopping import MaxItemStopper
        >>> from catsim.simulation import Simulator
        >>> from catsim.cat import generate_item_bank
        >>> initializer = RandomInitializer()
        >>> selector = MaxInfoSelector()
        >>> estimator = HillClimbingEstimator()
        >>> stopper = MaxItemStopper(20)
        >>> Simulator(generate_item_bank(100), 10).simulate(initializer, selector, estimator, stopper)
        """

        if initializer is not None:
            self._initializer = initializer
        if selector is not None:
            self._selector = selector
        if estimator is not None:
            self._estimator = estimator
        if stopper is not None:
            self._stopper = stopper

        for s in [self._initializer, self._selector, self._estimator, self._stopper]:
            s.simulator = self

        if verbose:
            print(
                (
                    'Starting simulation: {0} {1} {2} {3} {4} items'.format(
                        self._initializer, self._selector, self._estimator, self._stopper,
                        self._items.shape[0]
                    )
                )
            )
            pbar = tqdm.tqdm(total=len(self.examinees))

        start_time = time.time()

        for current_examinee, true_theta in enumerate(self.examinees):

            if verbose:
                pbar.update()

            est_theta = self._initializer.initialize(current_examinee)
            self._estimations[current_examinee].append(est_theta)

            while not self._stopper.stop(current_examinee):
                selected_item = self._selector.select(current_examinee)

                # if the selector returns None, it means the selector and not the stopper, is asking the test to stop
                # this happens e.g. if the item bank or or the available strata end before the minimum error is achieved
                if selected_item is None:
                    break

                # simulates the examinee's response via the four-parameter
                # logistic function
                response = irt.icc(
                    true_theta, self.items[selected_item][0], self.items[selected_item][1],
                    self.items[selected_item][2], self.items[selected_item][3]
                ) >= numpy.random.uniform()

                self._response_vectors[current_examinee].append(response)

                # adds the item selected by the selector to the pool of administered items
                self._administered_items[current_examinee].append(selected_item)

                # estimate the new theta using the given estimator
                est_theta = self._estimator.estimate(current_examinee)

                # count occurrences of this item in all tests
                item_occurrences = numpy.sum(
                    [
                        selected_item in administered_list
                        for administered_list in self._administered_items
                    ]
                )

                # update the exposure value for this item
                # r = number of tests item has been used on / total number of tests
                self.items[selected_item, 4] = item_occurrences / len(self.examinees)

                self._estimations[current_examinee].append(est_theta)

        self._duration = (time.time() - start_time)

        if verbose:
            pbar.close()
            print('Simulation took {0} seconds'.format(self._duration))

        self._bias = cat.bias(self.examinees, self.latest_estimations)
        self._mse = cat.mse(self.examinees, self.latest_estimations)
        self._rmse = cat.rmse(self.examinees, self.latest_estimations)

        # overlap is calculated only if all examinees answered the same amount of items
        # maybe there is a way to calculate it with tests of different lengths,
        # but I did not find it in the literature
        test_size = None
        len_first = len(self._administered_items[0]) if self._administered_items else None
        if type(selector) is FiniteSelector:
            test_size = selector.test_size
        elif all(len(i) == len_first for i in self._administered_items):
            test_size = len_first
        if test_size is not None:
            self._overlap_rate = cat.overlap_rate(self.items, test_size)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
