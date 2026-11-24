"""Tests for the ItemBank class."""

import copy
import math
import operator
import pickle
import re
from collections.abc import Callable

import numpy as np
import pytest

from catsim import IrtModel, ItemBank
from catsim import item_bank as item_bank_module


class TestItemBankCreation:
  """Tests for ItemBank initialization and creation."""

  @pytest.mark.parametrize(
    ("items", "expected"),
    [
      pytest.param([0.5], [[1.0, 0.5, 0.0, 1.0]], id="difficulty"),
      pytest.param([1.2, -0.3, 0.1, 0.9], [[1.2, -0.3, 0.1, 0.9]], id="four parameters"),
    ],
  )
  def test_a_one_dimensional_array_is_one_item(self, items: list[float], expected: list[list[float]]) -> None:
    """A 1-D array is one item's parameters; reading it as a column would make one item per value."""
    np.testing.assert_array_equal(ItemBank(np.array(items)).items, expected)

  def test_a_nested_list_raises_type_error(self) -> None:
    """A nested list is not an item matrix; without the type check it fails later with an unrelated AttributeError."""
    with pytest.raises(TypeError, match=r"not of type numpy\.ndarray"):
      ItemBank([[1.0, 0.0, 0.1, 0.95]])  # type: ignore[arg-type]

  @pytest.mark.parametrize("validate", [True, False])
  def test_rejects_a_matrix_with_more_than_four_columns(self, validate: bool) -> None:
    """A bank saved with the old fifth exposure-rate column raises and says how to drop it, instead of being misread."""
    items = np.array([[1.0, 0.0, 0.1, 0.95, 0.3]])

    with pytest.raises(ValueError, match=r"5 columns.*items\[:, :4\]"):
      ItemBank(items, validate=validate)

  @pytest.mark.parametrize("validate", [True, False])
  def test_rejects_a_matrix_without_columns(self, validate: bool) -> None:
    """An (n, 0) matrix raises at construction instead of failing at first use with an IndexError."""
    with pytest.raises(ValueError, match="0 columns"):
      ItemBank(np.empty((3, 0)), validate=validate)

  def test_keeps_its_own_float_copy_of_the_parameters(self) -> None:
    """Editing the caller's matrix later must not change a validated bank or leave its cached values stale.

    Integer input is stored as floats.
    """
    params = np.array([[1, 0, 0, 1], [2, 1, 0, 1]])
    bank = ItemBank(params)
    max_info_thetas = bank.max_info_thetas.copy()

    params[0, 0] = -5

    assert bank.items.dtype == np.float64
    assert bank.discrimination.tolist() == [1.0, 2.0]
    np.testing.assert_array_equal(bank.max_info_thetas, max_info_thetas)

  @pytest.mark.parametrize(
    ("view", "index"),
    [
      pytest.param(lambda bank: bank.items, (0, 1), id="items"),
      pytest.param(lambda bank: bank.difficulty, 0, id="difficulty"),
      pytest.param(lambda bank: bank.get_item(0), 1, id="get_item"),
      pytest.param(operator.itemgetter(0), 1, id="getitem"),
    ],
  )
  @pytest.mark.parametrize("validate", [True, False])
  def test_parameters_are_read_only(
    self, view: Callable[[ItemBank], np.ndarray], index: int | tuple[int, int], validate: bool
  ) -> None:
    """Writing into the parameters raises and leaves the bank alone, validated or not.

    The arrays used to be writable, so a write changed the difficulty behind the cached max_info_thetas, model and
    strata.
    """
    bank = ItemBank(np.array([[1.0, 0.5, 0.0, 1.0], [1.3, -0.2, 0.0, 1.0]]), validate=validate)

    with pytest.raises(ValueError, match="read-only"):
      view(bank)[index] = 99.0

    assert bank.difficulty.tolist() == [0.5, -0.2]

  @pytest.mark.parametrize(
    "cache",
    [
      pytest.param(lambda bank: bank.max_info_thetas, id="max_info_thetas"),
      pytest.param(lambda bank: bank.max_info_values, id="max_info_values"),
    ],
  )
  def test_max_info_caches_are_read_only(self, cache: Callable[[ItemBank], np.ndarray]) -> None:
    """Writing into a cached max-information array raises and leaves the cache alone.

    The stratified selectors rank items by these caches, so a writable cache let a caller change the ranking
    without touching the parameters. For 2PL items the maximum is at theta = b, where the information is a**2 / 4.
    """
    bank = ItemBank(np.array([[1.0, 0.5, 0.0, 1.0], [1.3, -0.2, 0.0, 1.0]]))

    with pytest.raises(ValueError, match="read-only"):
      cache(bank)[0] = 99.0

    assert bank.max_info_thetas.tolist() == [0.5, -0.2]
    assert bank.max_info_values.tolist() == pytest.approx([0.25, 0.4225])

  @pytest.mark.parametrize(
    "duplicate",
    [
      pytest.param(copy.deepcopy, id="deepcopy"),
      pytest.param(lambda bank: pickle.loads(pickle.dumps(bank)), id="pickle"),
    ],
  )
  @pytest.mark.parametrize(
    "array",
    [
      pytest.param(lambda bank: bank.items, id="items"),
      pytest.param(lambda bank: bank.max_info_thetas, id="max_info_thetas"),
      pytest.param(lambda bank: bank.max_info_values, id="max_info_values"),
    ],
  )
  def test_copies_stay_read_only(
    self, duplicate: Callable[[ItemBank], ItemBank], array: Callable[[ItemBank], np.ndarray]
  ) -> None:
    """A deep-copied or unpickled bank, e.g. one sent to a worker process, is as read-only as the original.

    NumPy does not keep the read-only flag through deepcopy or pickle, so without restoring it a copy's parameters
    and caches became writable again.
    """
    bank = ItemBank(np.array([[1.0, 0.5, 0.0, 1.0], [1.3, -0.2, 0.0, 1.0]]))
    bank.max_info_values  # noqa: B018 - fill both caches before copying, so the copy carries them

    copied = duplicate(bank)

    with pytest.raises(ValueError, match="read-only"):
      array(copied)[0] = 99.0

  def test_validation_enabled_by_default(self) -> None:
    """A bank is validated unless told otherwise, so a = -1 raises instead of being accepted."""
    items = np.array([[-1.0, 0.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="discrimination <= 0"):
      ItemBank(items)

  def test_validation_can_be_disabled(self) -> None:
    """With validate=False the parameter checks are skipped, so a bank with a = -1 is built instead of raising."""
    items = np.array([[-1.0, 0.0, 0.0, 1.0]])

    bank = ItemBank(items, validate=False)
    assert bank.n_items == 1


class TestItemBankValidation:
  """Tests for the parameter checks a bank runs when validate is True."""

  def test_accepts_parameters_on_the_edges_of_their_ranges(self) -> None:
    """A small positive a, c = 0 and d = 1 are valid; a strict comparison in the wrong place would reject them."""
    items = np.array([[1e-3, -5.0, 0.0, 1.0], [1.2, 0.5, 0.15, 0.98]])

    np.testing.assert_array_equal(ItemBank(items).items, items)

  @pytest.mark.parametrize(
    "shape",
    [
      pytest.param((4,), id="one dimension"),
      pytest.param((3, 0), id="no columns"),
      pytest.param((2, 3), id="three columns"),
      pytest.param((1, 5), id="five columns"),
      pytest.param((2, 4, 1), id="three dimensions"),
    ],
  )
  def test_rejects_a_matrix_that_is_not_n_by_4(self, shape: tuple[int, ...]) -> None:
    """A matrix that is not n x 4 raises ValueError naming its shape.

    Without the shape check, missing columns fail with an IndexError and a fifth column is ignored. ItemBank completes
    1 to 3 columns first, so only a matrix with more than two dimensions reaches this check through it.
    """
    with pytest.raises(ValueError, match=re.escape(f"shape {shape}; it must be 2-D with exactly 4 columns")):
      item_bank_module._validate(np.ones(shape))

  def test_rejects_a_matrix_without_items(self) -> None:
    """A 0 x 4 matrix raises ValueError: with no rows, every parameter check passes vacuously."""
    with pytest.raises(ValueError, match="no items"):
      ItemBank(np.empty((0, 4)))

  @pytest.mark.parametrize(
    ("item", "message"),
    [
      pytest.param([-0.5, 0.0, 0.1, 0.95], "discrimination <= 0", id="a < 0"),
      pytest.param([0.0, 0.0, 0.1, 0.95], "discrimination <= 0", id="a = 0"),
      pytest.param([1.0, 0.0, -0.1, 0.95], "pseudo-guessing < 0", id="c < 0"),
      pytest.param([1.0, 0.0, 1.5, 0.95], "pseudo-guessing > 1", id="c > 1"),
      pytest.param([1.0, 0.0, 0.1, 1.5], "upper asymptote > 1", id="d > 1"),
      pytest.param([1.0, 0.0, 0.1, -0.1], "upper asymptote < 0", id="d < 0"),
      pytest.param([1.0, 0.0, 0.3, 0.3], "pseudo-guessing >= upper asymptote", id="c = d"),
    ],
  )
  def test_rejects_a_parameter_out_of_its_range(self, item: list[float], message: str) -> None:
    """Each range check raises ValueError naming the parameter; printing the message and returning would accept it.

    a = 0 gives no information and no point of maximum information; c >= d makes the ICC flat or decreasing and the
    information formula divide by d - c.
    """
    items = np.array([[1.0, 0.0, 0.0, 1.0], item])

    with pytest.raises(ValueError, match=message):
      ItemBank(items)

  @pytest.mark.parametrize(
    "item",
    [
      pytest.param([np.nan, 0.0, 0.0, 1.0], id="a = nan"),
      pytest.param([1.0, np.nan, 0.0, 1.0], id="b = nan"),
      pytest.param([1.0, 0.0, np.nan, 1.0], id="c = nan"),
      pytest.param([1.0, 0.0, 0.0, np.nan], id="d = nan"),
      pytest.param([np.inf, 0.0, 0.0, 1.0], id="a = inf"),
      pytest.param([-np.inf, 0.0, 0.0, 1.0], id="a = -inf"),
      pytest.param([1.0, np.inf, 0.0, 1.0], id="b = inf"),
      pytest.param([1.0, -np.inf, 0.0, 1.0], id="b = -inf"),
    ],
  )
  def test_rejects_nan_and_infinite_parameters(self, item: list[float]) -> None:
    """NaN and infinite parameters raise ValueError.

    Every range check compares, and comparisons with NaN are false; without a finiteness check b is never checked and
    a = inf passes, so such an item turns the ICC, likelihoods and information sums it enters into NaN.
    """
    items = np.array([[1.0, 0.5, 0.1, 1.0], item])

    with pytest.raises(ValueError, match="NaN or infinite parameters"):
      ItemBank(items)

  def test_reports_every_violation_on_one_line(self) -> None:
    """All violations are listed, separated by "; "; starting with and joining by newlines would not match."""
    items = np.array([[-1.0, 0.0, -0.1, 1.0]])
    expected = "There are items with discrimination <= 0; There are items with pseudo-guessing < 0"

    with pytest.raises(ValueError, match=rf"\A{re.escape(expected)}\Z"):
      ItemBank(items)


class TestItemBankGenerate:
  """Tests for ItemBank.generate() factory method."""

  def test_generate_4pl(self) -> None:
    """Test generating 4PL item bank."""
    bank = ItemBank.generate(50, item_type="4PL", seed=42)

    assert bank.n_items == 50
    assert bank.model == IrtModel.PL4
    # Check distributions are reasonable
    assert 0.5 < np.mean(bank.discrimination) < 2.0
    assert -2 < np.mean(bank.difficulty) < 2
    assert np.min(bank.pseudo_guessing) >= 0
    assert 0.94 <= np.min(bank.upper_asymptote) <= 1.0

  def test_generate_3pl(self) -> None:
    """Test generating 3PL item bank."""
    bank = ItemBank.generate(30, item_type="3PL", seed=42)

    assert bank.n_items == 30
    assert bank.model == IrtModel.PL3
    assert np.allclose(bank.upper_asymptote, 1.0)
    assert np.any(bank.pseudo_guessing > 0)

  def test_generate_2pl(self) -> None:
    """Test generating 2PL item bank."""
    bank = ItemBank.generate(20, item_type="2PL", seed=42)

    assert bank.n_items == 20
    assert bank.model == IrtModel.PL2
    assert np.allclose(bank.pseudo_guessing, 0.0)
    assert np.allclose(bank.upper_asymptote, 1.0)

  def test_generate_1pl(self) -> None:
    """Test generating 1PL (Rasch) item bank."""
    bank = ItemBank.generate(15, item_type="1PL", seed=42)

    assert bank.n_items == 15
    assert bank.model == IrtModel.PL1
    assert np.allclose(bank.discrimination, 1.0)
    assert np.allclose(bank.pseudo_guessing, 0.0)
    assert np.allclose(bank.upper_asymptote, 1.0)

  def test_generate_with_correlation(self) -> None:
    """Test generating items with correlated a and b parameters."""
    bank = ItemBank.generate(100, item_type="2PL", corr=0.5, seed=42)

    # Check that there's positive correlation
    correlation = np.corrcoef(bank.discrimination, bank.difficulty)[0, 1]
    assert correlation > 0.3  # Should be somewhat positive

  def test_generate_case_insensitive(self) -> None:
    """Test that item_type strings are case insensitive."""
    bank1 = ItemBank.generate(10, item_type="3pl", seed=42)
    bank2 = ItemBank.generate(10, item_type="3PL", seed=42)
    bank3 = ItemBank.generate(10, item_type="3Pl", seed=42)

    assert bank1.model == bank2.model == bank3.model == IrtModel.PL3

  def test_generate_invalid_type_raises(self) -> None:
    """Test that invalid item_type raises ValueError."""
    with pytest.raises(ValueError, match="'5PL' is not a valid IrtModel"):
      ItemBank.generate(10, item_type="5PL")

  def test_generate_with_an_irt_model(self) -> None:
    """An IrtModel member is accepted like its name; a conversion that only takes plain strings would reject it."""
    bank = ItemBank.generate(10, item_type=IrtModel.PL3, seed=42)

    assert bank.model == IrtModel.PL3
    assert np.all(bank.pseudo_guessing > 0)
    np.testing.assert_array_equal(bank.upper_asymptote, 1.0)

  def test_generate_redraws_non_positive_discriminations(self) -> None:
    """A negative draw is redrawn rather than shifting every discrimination.

    With seed 876, one of the 500 drawn discriminations is negative. Earlier versions added its absolute
    value to every item, which left one item with a = 0 and moved the whole distribution.
    """
    rng = np.random.default_rng(876)
    _, raw_a = rng.multivariate_normal([0, 1.2], [[1, 0], [0, 0.25**2]], 500).T
    assert (raw_a <= 0).sum() == 1

    bank = ItemBank.generate_item_bank(500, itemtype="2PL", seed=876)

    assert bank.discrimination.min() > 0
    np.testing.assert_array_equal(bank.discrimination[raw_a > 0], raw_a[raw_a > 0])


class TestItemBankProperties:
  """Tests for ItemBank properties."""

  def test_discrimination_property(self) -> None:
    """Test discrimination property returns column 0."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert np.array_equal(bank.discrimination, np.array([1.0, 1.5]))

  def test_difficulty_property(self) -> None:
    """Test difficulty property returns column 1."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert np.array_equal(bank.difficulty, np.array([0.0, 1.0]))

  def test_pseudo_guessing_property(self) -> None:
    """Test pseudo_guessing property returns column 2."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert np.array_equal(bank.pseudo_guessing, np.array([0.0, 0.1]))

  def test_upper_asymptote_property(self) -> None:
    """Test upper_asymptote property returns column 3."""
    items = np.array([[1.0, 0.0, 0.0, 0.95], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert np.array_equal(bank.upper_asymptote, np.array([0.95, 1.0]))

  @pytest.mark.parametrize(
    ("items", "model"),
    [
      pytest.param([[1.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]], IrtModel.PL1, id="1PL"),
      pytest.param([[0.8, -1.0, 0.0, 1.0], [1.2, 0.0, 0.0, 1.0], [1.5, 1.0, 0.0, 1.0]], IrtModel.PL2, id="2PL"),
      pytest.param([[0.8, -1.0, 0.1, 1.0], [1.2, 0.0, 0.15, 1.0], [1.5, 1.0, 0.2, 1.0]], IrtModel.PL3, id="3PL"),
      pytest.param([[1.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.95]], IrtModel.PL4, id="4PL with c = 0"),
      pytest.param([[0.8, -1.0, 0.1, 0.95], [1.2, 0.0, 0.15, 0.98], [1.5, 1.0, 0.2, 0.99]], IrtModel.PL4, id="4PL"),
    ],
  )
  def test_model_is_the_simplest_that_fits_every_item(self, items: list[list[float]], model: IrtModel) -> None:
    """A bank is 1PL, 2PL, 3PL or 4PL by the parameters that vary; a wrong column, or checking c before d, misreads it.

    The model also equals the member's name, as a string.
    """
    detected = ItemBank(np.array(items)).model

    assert detected is model
    assert detected == model.value


class TestItemBankMethods:
  """Tests for ItemBank methods."""

  def test_get_item(self) -> None:
    """Test getting a single item by index."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    item = bank.get_item(1)
    assert len(item) == 4
    assert item[0] == pytest.approx(1.5)  # discrimination
    assert item[1] == pytest.approx(1.0)  # difficulty

  def test_get_item_out_of_bounds_raises(self) -> None:
    """Test that out of bounds index raises IndexError."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    with pytest.raises(IndexError, match="out of bounds"):
      bank.get_item(5)

    with pytest.raises(IndexError, match="out of bounds"):
      bank.get_item(-1)

  def test_get_items(self) -> None:
    """Test getting multiple items by indices."""
    items = np.array([
      [1.0, 0.0, 0.0, 1.0],
      [1.5, 1.0, 0.1, 1.0],
      [0.8, -0.5, 0.05, 1.0],
    ])
    bank = ItemBank(items)

    selected = bank.get_items([0, 2])
    assert selected.shape == (2, 4)
    assert selected[0, 0] == pytest.approx(1.0)
    assert selected[1, 0] == pytest.approx(0.8)

  @pytest.mark.parametrize(
    ("method", "expected"),
    [
      pytest.param(ItemBank.icc, [1 / (1 + math.exp(-0.5)), 0.5], id="icc"),
      pytest.param(ItemBank.information, [math.exp(-0.5) / (1 + math.exp(-0.5)) ** 2, 1.0], id="information"),
    ],
  )
  @pytest.mark.parametrize("theta", [np.float32(0.5), np.array(0.5), 0.5], ids=["float32", "0-d array", "float"])
  def test_a_scalar_theta_of_any_type_gives_one_value_per_item(
    self, method: Callable[[ItemBank, object], np.ndarray], expected: list[float], theta: object
  ) -> None:
    """A NumPy float32 or a 0-d array is one ability, like a float.

    An isinstance check for int and float sends them down the path for arrays of abilities, where len() raises
    TypeError. At theta = 0.5 the 1PL item at b = 0 has P = 1 / (1 + e^-0.5) and P (1 - P), the 2PL item with a = 2
    at its difficulty has P = 1/2 and a^2 / 4 = 1.
    """
    bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.5, 0.0, 1.0]]))

    values = method(bank, theta)

    assert values.shape == (2,)
    np.testing.assert_allclose(values, expected, rtol=1e-12)

  @pytest.mark.parametrize("method", [ItemBank.icc, ItemBank.information], ids=["icc", "information"])
  def test_k_abilities_give_one_row_per_ability(self, method: Callable[[ItemBank, object], np.ndarray]) -> None:
    """Three abilities and two items give a 3 x 2 array whose row j holds the values at ability j.

    Pairing ability j with item j element-wise raises for three abilities and two items.
    """
    bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]]))
    thetas = np.array([-1.0, 0.0, 1.0])

    values = method(bank, thetas)

    assert values.shape == (3, 2)
    np.testing.assert_allclose(values, [method(bank, float(theta)) for theta in thetas], rtol=1e-12)

  def test_test_information_subset_items(self) -> None:
    """Only the listed items count: items 0 and 2 give 1/4 + L (1 - L) at theta = 0, with L = 1 / (1 + e^-1).

    Summing every item adds another L (1 - L), for item 1.
    """
    bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]]))
    logistic = 1 / (1 + math.exp(-1.0))

    assert bank.test_information(0.0, item_indices=[0, 2]) == pytest.approx(0.25 + logistic * (1 - logistic), rel=1e-12)


class TestItemBankDunderMethods:
  """Tests for ItemBank dunder methods."""

  def test_getitem_int(self) -> None:
    """Test __getitem__ with integer index."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    item = bank[1]
    assert item[0] == pytest.approx(1.5)

  def test_getitem_slice(self) -> None:
    """Test __getitem__ with slice."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0], [0.8, -0.5, 0.0, 1.0]])
    bank = ItemBank(items)

    subset = bank[0:2]
    assert subset.shape == (2, 4)

  def test_getitem_list(self) -> None:
    """Test __getitem__ with list of indices."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0], [0.8, -0.5, 0.0, 1.0]])
    bank = ItemBank(items)

    subset = bank[[0, 2]]
    assert subset.shape == (2, 4)
    assert subset[0, 0] == pytest.approx(1.0)
    assert subset[1, 0] == pytest.approx(0.8)
