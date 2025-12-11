"""Tests for catsim.exceptions module."""

import pytest

from catsim.exceptions import NoItemsAvailableError


class TestNoItemsAvailableError:
  """Tests for NoItemsAvailableError exception."""

  def test_is_runtime_error(self) -> None:
    """Test that NoItemsAvailableError inherits from RuntimeError."""
    assert issubclass(NoItemsAvailableError, RuntimeError)

  def test_can_be_raised(self) -> None:
    """Test that the exception can be raised."""
    msg = "No items available"
    with pytest.raises(NoItemsAvailableError):
      raise NoItemsAvailableError(msg)

  def test_message_is_preserved(self) -> None:
    """Test that the error message is preserved."""
    message = "Custom error message"
    try:
      raise NoItemsAvailableError(message)
    except NoItemsAvailableError as e:
      assert str(e) == message

  def test_can_be_caught_as_runtime_error(self) -> None:
    """Test that it can be caught as RuntimeError."""
    msg = "No items available"
    with pytest.raises(RuntimeError):
      raise NoItemsAvailableError(msg)
