"""Property-based tests shared by every item selector."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import BaseSelector, ClusterSelector
from tests.conftest import SELECTOR_FACTORIES, TEST_SIZE
from tests.contexts import make_context
from tests.hypothesis_strategies import item_params_matrix

SEL_SETTINGS = settings(max_examples=25, deadline=None)
THETAS = st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False)
SEEDS = st.integers(min_value=0, max_value=2**31 - 1)


def _session(
  selector: BaseSelector, item_bank: ItemBank, responses: list[bool], *, theta: float, seed: int
) -> list[int]:
  """Return the items ``selector`` picks for a session of ``len(responses)`` items, each pick taken by the next.

  Stratified selectors tie the test position to a stratum, so the administered items must be the selector's own
  earlier picks: an arbitrary set can exhaust the current stratum.
  """
  rng = np.random.default_rng(seed)
  picks: list[int] = []
  for answered in range(len(responses)):
    picks.append(selector.select(make_context(item_bank, picks, responses[:answered], theta=theta, rng=rng)))
  return picks


@composite
def bank_and_responses(draw: DrawFn, *, picks_left: int) -> tuple[ItemBank, list[bool]]:
  """An item bank and the responses of a session that leaves at least ``picks_left`` of its items unpicked.

  The session is never longer than TEST_SIZE, the test length the stratified selectors of SELECTOR_FACTORIES are
  built for.
  """
  item_bank = ItemBank(draw(item_params_matrix(min_items=2, max_items=12)))
  length = draw(st.integers(min_value=0, max_value=min(TEST_SIZE, item_bank.n_items) - picks_left))
  return item_bank, draw(st.lists(st.booleans(), min_size=length, max_size=length))


@pytest.mark.parametrize("name", sorted(SELECTOR_FACTORIES))
@given(bank_responses=bank_and_responses(picks_left=0), theta=THETAS, seed=SEEDS)
@SEL_SETTINGS
def test_every_pick_is_an_unseen_builtin_int_of_the_bank(
  name: str, bank_responses: tuple[ItemBank, list[bool]], theta: float, seed: int
) -> None:
  """Each pick of a session indexes the bank, as a builtin int, and repeats no earlier pick.

  A selector that drew from the whole bank instead of the remaining items, returned an index past the bank or let a
  numpy integer into the session state fails.
  """
  item_bank, responses = bank_responses

  picks = _session(SELECTOR_FACTORIES[name](item_bank.n_items), item_bank, responses, theta=theta, seed=seed)

  assert all(type(item_id) is int for item_id in picks)
  assert all(0 <= item_id < item_bank.n_items for item_id in picks)
  assert len(set(picks)) == len(picks)


@pytest.mark.parametrize("name", sorted(SELECTOR_FACTORIES))
@given(bank_responses=bank_and_responses(picks_left=1), theta=THETAS, seed=SEEDS)
@SEL_SETTINGS
def test_select_raises_once_the_bank_is_administered_or_excluded(
  name: str, bank_responses: tuple[ItemBank, list[bool]], theta: float, seed: int
) -> None:
  """With every item the session did not pick excluded, select raises NoItemsAvailableError, which ends the session.

  A selector that scored an empty candidate array, handed back an administered or excluded item, or failed with
  another error fails.
  """
  item_bank, responses = bank_responses
  selector = SELECTOR_FACTORIES[name](item_bank.n_items)
  picks = _session(selector, item_bank, responses, theta=theta, seed=seed)
  context = make_context(
    item_bank,
    picks,
    responses,
    theta=theta,
    excluded_items=sorted(set(range(item_bank.n_items)) - set(picks)),
  )

  with pytest.raises(NoItemsAvailableError, match="no more items"):
    selector.select(context)


@given(
  params=item_params_matrix(min_items=4, max_items=16),
  data=st.data(),
  method=st.sampled_from(["item_info", "cluster_info", "weighted_info"]),
  theta=THETAS,
  seed=SEEDS,
)
@SEL_SETTINGS
def test_cluster_picks_are_unseen_items_of_the_bank_for_any_labels(
  params: np.ndarray, data: st.DataObject, method: str, theta: float, seed: int
) -> None:
  """Under every method, with labels that skip values and so leave clusters with no items, picks stay valid.

  A method that scored an empty cluster, or picked from a cluster whose items were all administered, would return an
  administered item or an index past the bank.
  """
  item_bank = ItemBank(params)
  n_clusters = data.draw(st.integers(min_value=1, max_value=item_bank.n_items // 2))
  clusters = data.draw(st.lists(st.integers(0, n_clusters - 1), min_size=item_bank.n_items, max_size=item_bank.n_items))
  responses = data.draw(st.lists(st.booleans(), max_size=item_bank.n_items))

  picks = _session(ClusterSelector(clusters=clusters, method=method), item_bank, responses, theta=theta, seed=seed)

  assert all(0 <= item_id < item_bank.n_items for item_id in picks)
  assert len(set(picks)) == len(picks)
