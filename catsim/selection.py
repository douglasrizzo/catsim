from catsim import irt
import numpy as np
from abc import ABCMeta, abstractmethod


class Selector:
    """Base class representing a CAT item selector"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def select(self):
        pass


class MaxInfoSelector(Selector):
    """Selector that returns the first non-administered item with maximum information, given an estimated theta"""

    def __init__(self, items, **kwargs):
        super().__init__(items)

    def select(items, administered_items, est_theta):
        """Get the indexes of all items that have not yet been administered, calculate
        their information value and pick the one with maximum information

        :param items: an nx3 item matrix
        :type items: numpy.ndarray
        :param administered_items: a list with the indexes of all administered from the item matrix
        :param est_theta: estimated proficiency value
        :type est_theta: float
        :returns: index of the first non-administered item with maximum information
        :rtype: int
        """
        valid_indexes = np.array(
            list(set(range(items.shape[0])) - set(administered_items)))

        inf_values = [irt.inf(est_theta, i[0], i[1], i[2])
                      for i in items[valid_indexes]]

        valid_indexes = [
            index for (inf_value, index) in sorted(zip(inf_values, valid_indexes), reverse=True)]

        return valid_indexes[0]


class ClusterSelector(Selector):
    """Cluster-based Item Selection Method

        .. [Men15] Meneghetti, D. R. (2015). Metolodogia de seleção de itens em testes adaptativos informatizados baseada em agrupamento por similaridade (Mestrado). Centro Universitário da FEI. Retrieved from https://www.researchgate.net/publication/283944553_Metodologia_de_selecao_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade
    """
    pass
