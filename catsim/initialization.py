import numpy

from catsim.simulation import Initializer


class RandomInitializer(Initializer):
    """Randomly initializes the first estimate of an examinee's proficiency

    :param dist_type: either `uniform` or `normal`
    :param dist_params: a tuple containing minimum and maximum values for the
                        uniform distribution (in no particular order) or the average
                        and standard deviation values for the normal distribution
                        (in this particular order)."""

    def __str__(self):
        return 'Random Initializer'

    def __init__(self, dist_type: str = 'uniform', dist_params: tuple = (-5, 5)):
        super(RandomInitializer, self).__init__()

        available_distributions = ['uniform', 'normal']
        if dist_type not in available_distributions:
            raise ValueError('{0} not in available distributions {1}'.format(dist_type, available_distributions))

        self._dist_type = dist_type
        self._dist_params = dist_params

    def initialize(self, index: int = None, **kwargs) -> float:
        """Generates a value using the chosen distribution and parameters

        :param index: the index of the current examinee. This parameter is not used by this method.
        :returns: a proficiency value generated from the chosen distribution using the passed parameters"""
        if self._dist_type == 'uniform':
            return numpy.random.uniform(min(self._dist_params), max(self._dist_params))
        elif self._dist_type == 'normal':
            return numpy.random.normal(self._dist_params[0], self._dist_params[1])


class FixedPointInitializer(Initializer):
    """Initializes every proficiency at the same point."""

    def __str__(self):
        return 'Fixed Point Initializer'

    def __init__(self, start: float):
        """
        :param start: the starting point for every examinee
        """
        super(FixedPointInitializer, self).__init__()
        self._start = start

    def initialize(self, index: int = None, **kwargs) -> float:
        """Returns the same proficiency value that was passed to the constructor of the initializer

        :param index: the index of the current examinee. This parameter is not used by this method.
        :returns: the same proficiency value that was passed to the constructor of the initializer"""
        return self._start
