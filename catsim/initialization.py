from abc import ABCMeta, abstractmethod
import numpy


class Initializer:
    """Base class for CAT initializers"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Initializer, self).__init__()

    @abstractmethod
    def initialize(self):
        pass


class RandomInitializer(Initializer):
    """Randomly initializes the first estimate of an examinee's proficiency"""

    def __init__(self, arg):
        super(RandomInitializer, self).__init__()
        self.arg = arg

    def initialize():
        return numpy.random.uniform(-5, 5)
