"""Custom exceptions for the catsim package."""


class NoItemsAvailableError(RuntimeError):
  """Exception raised by an item selector that has no item to offer.

  A selector raises it when none of the items it may choose from is left. That is not necessarily
  the end of the item bank: a selector that follows a fixed list of items or works through strata
  can run out first. ``CatEngine`` ends the session with ``StopReason.NO_ITEMS``.
  """
