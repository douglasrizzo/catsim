"""Random selector implementations."""

from numpy.random import Generator

from .._typing import FloatArray
from ..item_bank import ItemBank
from .base import BaseSelector, best_pool, remaining, require_rng


class RandomSelector(BaseSelector):
  """Selector that randomly selects items for application.

  This selector is useful for baseline comparisons or for studying the impact of
  item selection strategies.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Random Selector"

  def select(  # noqa: PLR6301
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,  # noqa: ARG002
    rng: Generator | None = None,
    exposure_rates: FloatArray | None = None,  # noqa: ARG002
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        Current estimated ability. Unused by this selector.
    rng : numpy.random.Generator or None, optional
        Random number generator the item is drawn with. Required.
    exposure_rates : numpy.ndarray or None, optional
        Ignored: this selector applies no exposure cap.
    response_vector : list[bool] or None, optional
        Unused; accepted for the engine's selector contract.

    Returns
    -------
    int
        Index of the next item to be applied, drawn uniformly at random from the items not yet
        administered.

    Raises
    ------
    ValueError
        If ``rng`` is None.
    NoItemsAvailableError
        If every item has already been administered.
    """
    return int(require_rng(rng).choice(remaining(item_bank.n_items, administered_items)))


class The54321Selector(BaseSelector):
  r"""Implementation of the 5-4-3-2-1 selector proposed by :cite:p:`McBride83`.

  The first item is drawn at random from the 5 most informative items at the current :math:`\hat\theta`,
  the second from the 4 most informative remaining items, and so on, so that from the fifth item on the
  most informative item is administered. Randomizing the opening items, which every examinee would
  otherwise share while :math:`\hat\theta` is still rough, reduces their exposure at little cost in
  precision. Items tied with the least informative item of a bin also join it, so from the fifth
  item on, ties for the most information are broken uniformly at random (ADR-0008). The selector
  applies no exposure cap: ``exposure_rates`` is ignored.

  Parameters
  ----------
  initial_bin_size : int, optional
      Number of most informative items the first item is drawn from; each following item draws from
      one fewer. Default is 5.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/the-54321-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "5-4-3-2-1 Selector"

  def __init__(self, initial_bin_size: int = 5) -> None:
    """Initialize a The54321Selector object.

    Parameters
    ----------
    initial_bin_size : int, optional
        Number of most informative items the first item is drawn from. Default is 5.

    Raises
    ------
    ValueError
        If ``initial_bin_size`` is smaller than 1.
    """
    if initial_bin_size < 1:
      msg = f"initial_bin_size must be at least 1, got {initial_bin_size}"
      raise ValueError(msg)
    super().__init__()
    self._initial_bin_size = initial_bin_size

  @property
  def initial_bin_size(self) -> int:
    """Get the number of most informative items the first item is drawn from."""
    return self._initial_bin_size

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: Generator | None = None,
    exposure_rates: FloatArray | None = None,  # noqa: ARG002
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator the item is drawn from the bin with. Required.
    exposure_rates : numpy.ndarray or None, optional
        Ignored: this selector applies no exposure cap.
    response_vector : list[bool] or None, optional
        Unused; accepted for the engine's selector contract.

    Returns
    -------
    int
        Index of the next item to be applied, drawn uniformly from the bin, which also holds every
        item tied with its least informative item (ADR-0008).

    Raises
    ------
    ValueError
        If ``rng`` is None.
    NoItemsAvailableError
        If every item has already been administered.
    """
    rng = require_rng(rng)

    candidates = remaining(item_bank.n_items, administered_items)
    bin_size = max(1, self._initial_bin_size - len(administered_items))
    # items tied with the bin's least informative item join the bin (ADR-0008)
    pool = best_pool(candidates, item_bank.information(est_theta)[candidates], bin_size)
    return int(rng.choice(pool))


class RandomesqueSelector(BaseSelector):
  """Implementation of the randomesque selector proposed by :cite:p:`Kingsbury89`.

  In this selector, at each step of the test, an item is randomly chosen from the :math:`n` most informative items in
  the item bank, :math:`n` being a user-defined value. Items tied with the least informative of those :math:`n` also
  join the bin (ADR-0008). The selector applies no exposure cap: ``exposure_rates`` is ignored.

  Parameters
  ----------
  bin_size : int
      Number of most informative remaining items each item is drawn from, at least 1.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Randomesque Selector"

  def __init__(self, bin_size: int) -> None:
    """Initialize a RandomesqueSelector object.

    Parameters
    ----------
    bin_size : int
        The number of most informative items to be taken into consideration when
        randomly selecting one of them.

    Raises
    ------
    ValueError
        If ``bin_size`` is smaller than 1.
    """
    if bin_size < 1:
      msg = f"bin_size must be at least 1, got {bin_size}"
      raise ValueError(msg)
    super().__init__()
    self._bin_size = bin_size

  @property
  def bin_size(self) -> int:
    """Get the bin size."""
    return self._bin_size

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: Generator | None = None,
    exposure_rates: FloatArray | None = None,  # noqa: ARG002
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator the item is drawn from the bin with. Required.
    exposure_rates : numpy.ndarray or None, optional
        Ignored: this selector applies no exposure cap.
    response_vector : list[bool] or None, optional
        Unused; accepted for the engine's selector contract.

    Returns
    -------
    int
        Index of the next item to be applied, drawn uniformly from the bin, which also holds every
        item tied with its least informative item (ADR-0008).

    Raises
    ------
    ValueError
        If ``rng`` is None.
    NoItemsAvailableError
        If every item has already been administered.
    """
    rng = require_rng(rng)

    candidates = remaining(item_bank.n_items, administered_items)
    # items tied with the bin's least informative item join the bin (ADR-0008)
    pool = best_pool(candidates, item_bank.information(est_theta)[candidates], self._bin_size)
    return int(rng.choice(pool))
