import math

def tpm(theta, a, b, c):
    """
    Item Response Theory three-parameter logistic function:

    .. math:: P(X_i = 1| \\theta) = c_i + \\frac{1-c_i}{1+ e^{Da_i(\\theta-b_i)}}

    Parameters
    ----------
    theta : float the individual's proficiency value. This parameter value has
            no boundary, but if a distribution of the form :math:`N(0, 1)` was
            used to estimate the parameters, then :math:`-4 \\leq \\theta \\leq
            4`.

    a : float the discrimination parameter of the item, usually a positive value
        in which :math:`0.8 \\leq a \\leq 2.5`.

    b : float the item difficulty parameter. This parameter value has no
        boundary, but if a distribution of the form :math:`N(0, 1)` was used to
        estimate the parameters, then :math:`-4 \\leq b \\leq 4`.

    c : float the item pseudo-guessing parameter. Being a probability,
        :math:`0\\leq c \\leq 1`, but items considered good usually have
        :math:`c \\leq 0.2`.
    """
    return c + ((1 - c) / (1 + math.exp(-a * (theta - b))))


def inf(theta, a, b, c):
    """
    Item Response Theory three-parameter information function

    .. math:: I(\\theta) = a^2\\frac{(P(\\theta)-c)^2}{(1-c)^2}.\\frac{(1-P(\\theta))}{P(\\theta)}

    Parameters
    ----------
    theta : float the individual's proficiency value. This parameter value has
            no boundary, but if a distribution of the form :math:`N(0, 1)` was
            used to estimate the parameters, then :math:`-4 \\leq \\theta \\leq
            4`.

    a : float the discrimination parameter of the item, usually a positive value
        in which :math:`0.8 \\leq a \\leq 2.5`.

    b : float the item difficulty parameter. This parameter value has no
        boundary, but if a distribution of the form :math:`N(0, 1)` was used to
        estimate the parameters, then :math:`-4 \\leq b \\leq 4`.

    c : float the item pseudo-guessing parameter. Being a probability,
        :math:`0\\leq c \\leq 1`, but items considered good usually have
        :math:`c \\leq 0.2`.
    """
    ml3 = tpm(theta, a, b, c)
    return math.pow(a, 2) * (math.pow(ml3 - c, 2) /
                             math.pow(1 - c, 2)) * (1 - ml3) / ml3