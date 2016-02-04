"""Class representing a method to apply a computerized adaptive test.
It contains methods representing the start conditions of the CAT,
reestimation of examinee's habilities after each item is
answered and the stopping rule for the CAT."""

from abc import ABCMeta, abstractmethod


class Procedure:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs): pass

    @abstractmethod
    def check_bank(self): pass

    @abstractmethod
    def next(self): pass

    @abstractmethod
    def start(self): pass

    @abstractmethod
    def stop(self): pass
