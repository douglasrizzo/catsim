from abc import ABCMeta, abstractmethod
import numpy


class Initializer:
    """Base class for CAT initializers"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Initializer, self).__init__()

    @abstractmethod
    def initialize(self):
        """Selects an examinee's initial :math:`\\theta` value

        :returns: examinee's initial :math:`\\theta` value
        :rtype: float
        """
        pass


class RandomInitializer(Initializer):
    """Randomly initializes the first estimate of an examinee's proficiency"""

    def __init__(self, dist_type='uniform', dist_params=(-5, 5)):
        super(RandomInitializer, self).__init__()

        available_distributions = ['uniform', 'normal']
        if dist_type not in available_distributions:
            raise ValueError(
                '{0} not in available distributions {1}'.format(
                    dist_type, available_distributions
                )
            )

        self.__dist_type = dist_type
        self.__dist_params = dist_params

    def initialize(self):
        if self.__dist_type == 'uniform':
            return numpy.random.uniform(min(self.__dist_params), max(self.__dist_params))
        elif self.__dist_type == 'normal':
            return numpy.random.normal(self.__dist_params[0], self.dist_params[1])
