"""Custom exceptions for the catsim package."""


class NoItemsAvailableError(RuntimeError):
  """Exception raised when no items are available for selection.

  This exception is raised by item selectors when they are unable to select
  a new item because all items in the item bank have been exhausted or
  no items meet the selection criteria.
  """
