from abc import ABCMeta, abstractmethod


class Stopper:
    """Base class for CAT stop criterion"""
    __metaclass__ = ABCMeta

    def __init__(self, arg):
        super(Stopper, self).__init__()
        self.arg = arg

    @abstractmethod
    def stop():
        pass


class MaxItemStopper(Stopper):
    """docstring for MaxItemStopper"""

    def __init__(self, arg):
        super(MaxItemStopper, self).__init__()
        self.arg = arg

    def stop(**kwargs):
        n_itens, max_itens = kwargs['n_itens', 'max_itens']
        if n_itens > max_itens:
            raise ValueError('More items than permitted were administered: {0} > {1}'.format(n_itens, max_itens))
