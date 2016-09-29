"""Module containing functions relevant to the process of simulating the
application of adaptive tests. Most of this module is based on the work of
[Bar10]_."""

import time

import numpy

from catsim import cat, irt
from catsim.estimation import Estimator
from catsim.initialization import Initializer
from catsim.selection import Selector
from catsim.stopping import Stopper


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
        initializer: Initializer=None,
        selector: Selector=None,
        estimator: Estimator=None,
        stopper: Stopper=None
    ):
        irt.validate_item_bank(items)

        # adds a column for each item's exposure rate and their cluster membership
        items = numpy.append(items, numpy.zeros([items.shape[0], 1]), axis=1)

        self._duration = 0
        self._items = items
        self._estimations = []
        self._administered_items = []

        self._initializer = initializer
        self._selector = selector
        self._estimator = estimator
        self._stopper = stopper

        # `examinees` is passed to its special setter
        self.examinees = examinees

    @property
    def items(self) -> numpy.ndarray:
        """Item matrix used by the simulator. If the simulation already
        occurred, a column containin item esposure rates will be added to the
        matrix."""
        return self._items

    @property
    def administered_items(self) -> list:
        """List of lists containin the indexes of items administered to each
        examinee during the simulation."""
        return self._administered_items

    @property
    def all_estimations(self) -> list:
        """List of lists containing all estimated :math:`\\hat\\theta` values
        for all examinees during each step of the test."""
        return self._estimations

    @property
    def estimations(self) -> list:
        """Final estimated :math:`\\hat\\theta` values for all examinees."""
        return [ests[-1] for ests in self._estimations]

    @property
    def examinees(self) -> list:
        """List containing examinees true proficiency values (:math:`\\theta`)."""
        return self._examinees

    @property
    def duration(self) -> float:
        """Duration of the simulation, in milliseconds."""
        return self._duration

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
        """Bias between the estimated and true proficiencies. This property is only available after :py:func:`simulate` has been successfully called. For more information on estimation bias, see :py:func:`catsim.cat.bias`"""
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
        else:
            raise ValueError('Examinees must be an int or list')

    def simulate(
        self,
        initializer: Initializer=None,
        selector: Selector=None,
        estimator: Estimator=None,
        stopper: Stopper=None,
        verbose: bool=False
    ):
        """Simulates a computerized adaptive testing application to one or more examinees

        :param initializer: an initializer that selects examinees :math:`\\theta_0`
        :param selector: a selector that selects new items to be presented to examinees
        :param estimator: an estimator that reestimates examinees proficiencies after each item is applied
        :param stopper: an object with a stopping criteria for the test
        :param verbose: whether to periodically print a message regarding the progress of the simulation. Good for longer simulations.

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

        if verbose:
            print(
                (
                    'Starting simulation: {0} {1} {2} {3}'.format(
                        self._initializer.__class__, selector.__class__, self._estimator.__class__,
                        self._stopper.__class__
                    )
                )
            )

        start_time = int(round(time.time() * 1000))
        for current_examinee, true_theta in enumerate(self.examinees):

            if verbose:
                print(('{0}/{1} examinees...'.format(current_examinee + 1, len(self.examinees))))

            est_theta = self._initializer.initialize()
            response_vector, administered_items, est_thetas = [], [], []

            while not self._stopper.stop(self.items[administered_items], est_thetas):
                try:
                    selected_item = self._selector.select(self.items, administered_items, est_theta)
                except:
                    print((len(administered_items)))
                    raise

                # simulates the examinee's response via the four-parameter
                # logistic function
                response = irt.icc(
                    true_theta, self.items[selected_item][0], self.items[selected_item][1],
                    self.items[selected_item][2], self.items[selected_item][3]
                ) >= numpy.random.uniform()

                response_vector.append(response)

                # adds the item selected by the selector to the pool of administered items
                administered_items.append(selected_item)

                # estimate the new theta using the given estimator
                est_theta = self._estimator.estimate(
                    response_vector, self.items[administered_items], est_theta
                )

                # flatten the list of lists so that we can count occurrences of items easier
                flattened_administered_items = [
                    administered_item
                    for administered_list in self.administered_items
                    for administered_item in administered_list
                ]

                # update the exposure value for this item
                # r = number of tests item has been used / total number of tests
                self.items[selected_item, 4] = numpy.sum(
                    flattened_administered_items == selected_item
                ) / len(
                    self.examinees
                )

                est_thetas.append(est_theta)

            self._estimations.append(est_thetas)
            self._administered_items.append(administered_items)

        self._duration = int(round(time.time() * 1000)) - start_time

        if verbose:
            print('Simulation took {0} milliseconds'.format(self._duration))

        self._bias = cat.bias(self.examinees, self.estimations)
        self._mse = cat.mse(self.examinees, self.estimations)
        self._rmse = cat.rmse(self.examinees, self.estimations)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
