"""Module containing functions relevant to the process of simulating the
application of adaptive tests. Most of this module is based on the work of
[Bar10]_.

.. [Bar10] BARRADA, Juan Ramón et al. A method for the comparison of item
   selection rules in computerized adaptive testing. Applied Psychological
   Measurement, v. 34, n. 6, p. 438-452, 2010."""

import numpy
from catsim import irt, cat
from catsim.initialization import Initializer
from catsim.selection import Selector
from catsim.reestimation import Estimator
from catsim.stopping import Stopper


class Simulator:
    """Class representing the simulator. It gathers several objects that describe the full
    simulation process and simulates one or more computerized adaptive tests

    :param items: a matrix containing item parameters
    :param examinees: an integer with the number of examinees, whose real :math:`\\theta` values will be
                      sampled from a normal distribution; or a :py:type:list containing said
                      :math:`\\theta_0` values
    """

    def __init__(self, items: numpy.ndarray, examinees):
        irt.validate_item_bank(items)

        # adds a column for each item's exposure rate and their cluster membership
        items = numpy.append(items, numpy.zeros([items.shape[0], 1]), axis=1)

        self.__items = items
        self.__estimations = []
        self.__administered_items = []

        # `examinees` is passed to its special setter
        self.examinees = examinees

    @property
    def items(self):
        return self.__items

    @property
    def administered_items(self):
        return self.__administered_items

    @property
    def estimations(self):
        return self.__estimations

    @property
    def examinees(self):
        return self.__examinees

    @examinees.setter
    def examinees(self, examinees):
        if type(examinees) == int:
            self.__examinees = numpy.random.normal(0, 1, examinees)
        elif type(examinees) == list:
            self.__examinees = numpy.array(examinees)

    def simulate(
        self, initializer: Initializer, selector: Selector, estimator: Estimator, stopper: Stopper
    ):
        """Simulates a computerized adaptive testing application to one or more examinees

        :param initializer: an initializer that selects examinees :math:`\\theta_0`
        :param selector: a selector that selects new items to be presented to examinees
        :param estimator: an estimator that reestimates examinees proficiencies after each item is applied
        :param stopper: an object with a stopping criteria for the test

        >>> from catsim.initialization import RandomInitializer
        >>> from catsim.selection import MaxInfoSelector
        >>> from catsim.reestimation import HillClimbingEstimator
        >>> from catsim.stopping import MaxItemStopper
        >>> from catsim.cat import generate_item_bank
        >>> initializer = RandomInitializer()
        >>> selector = MaxInfoSelector()
        >>> estimator = HillClimbingEstimator()
        >>> stopper = MaxItemStopper(20)
        >>> Simulator(generate_item_bank(100), 10).simulate(initializer, selector, estimator, stopper)
        """

        for true_theta in self.examinees:
            est_theta = initializer.initialize()
            response_vector, administered_items, est_thetas = [], [], []

            while not stopper.stop(self.items[administered_items], est_thetas):
                try:
                    selected_item = selector.select(self.items, administered_items, est_theta)
                except:
                    print(len(administered_items))
                    raise

                # simulates the examinee's response via the three-parameter
                # logistic function
                response = irt.tpm(
                    true_theta, self.items[selected_item][0], self.items[selected_item][1],
                    self.items[selected_item][2]
                ) >= numpy.random.uniform()

                response_vector.append(response)
                # adds the administered item to the pool of administered items
                administered_items.append(selected_item)

                if len(set(response_vector)) == 1:
                    est_theta = cat.dodd(est_theta, self.items, response)
                else:
                    est_theta = estimator.estimate(
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
                self.items[selected_item, 3] = numpy.sum(
                    flattened_administered_items == selected_item
                ) / len(self.examinees)

                est_thetas.append(est_theta)

            self.__estimations.append(est_thetas)
            self.__administered_items.append(administered_items)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
