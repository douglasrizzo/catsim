from abc import ABCMeta, abstractmethod


class Stopper:
    """Base class for CAT stop criterion"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Stopper, self).__init__()

    @abstractmethod
    def stop(self):
        pass


class MaxItemStopper(Stopper):
    """docstring for MaxItemStopper"""

    def __init__(self, max_itens):
        super(MaxItemStopper, self).__init__()
        self.__max_itens = max_itens

    def stop(self, n_itens: int) -> bool:
        """orelha"""
        if n_itens > self.__max_itens:
            raise ValueError('More items than permitted were administered: {0} > {1}'.format(n_itens, self.__max_itens))
