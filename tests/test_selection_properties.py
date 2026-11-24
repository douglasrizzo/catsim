"""Property-based tests shared by every item selector."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import BaseSelector, ClusterSelector
from tests.conftest import HALF_BANK_SELECTORS, SELECTOR_FACTORIES
from tests.contexts import make_context
from tests.hypothesis_strategies import item_params_matrix

SEL_SETTINGS = settings(max_examples=25)
BANKS = item_params_matrix(min_items=2, max_items=12)
THETAS = st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False)
SEEDS = st.integers(min_value=0, max_value=2**31 - 1)


def _session(
  selector: BaseSelector, item_bank: ItemBank, data: st.DataObject, *, seed: int, excluded: Sequence[int] = ()
) -> list[int]:
  """Return the items a session serves, each pick administered before the next, until the selector has none left.

  Every step draws a new ability estimate and response, so the next pick is not simply the next item of a ranking
  fixed at the start. Each pick must be a builtin int indexing the bank, neither administered nor excluded; the session
  ends when select raises NoItemsAvailableError, and any other error propagates.
  """
  rng = np.random.default_rng(seed)
  picks: list[int] = []
  responses: list[bool] = []
  while True:
    context = make_context(item_bank, picks, responses, theta=data.draw(THETAS), rng=rng, excluded_items=excluded)
    try:
      item_id = selector.select(context)
    except NoItemsAvailableError:
      return picks
    assert type(item_id) is int
    assert 0 <= item_id < item_bank.n_items
    assert item_id not in picks
    assert item_id not in excluded
    picks.append(item_id)
    responses.append(data.draw(st.booleans()))


@pytest.mark.parametrize("name", sorted(SELECTOR_FACTORIES))
@given(params=BANKS, data=st.data(), seed=SEEDS)
@SEL_SETTINGS
def test_a_session_serves_new_items_until_the_selector_runs_out(
  name: str, params: np.ndarray, data: st.DataObject, seed: int
) -> None:
  """A session serves distinct items of the bank, as builtin ints, then ends with NoItemsAvailableError.

  The linear and stratified selectors of SELECTOR_FACTORIES run out after half the bank, rounded up, through their own
  checks (the list, the stratum); every other selector serves the whole bank. A selector that ignored the administered
  items, returned an index past the bank or a numpy integer, kept serving once out of items, raised another error, or
  ran out early fails.
  """
  item_bank = ItemBank(params)

  picks = _session(SELECTOR_FACTORIES[name](item_bank.n_items), item_bank, data, seed=seed)

  assert len(picks) == ((item_bank.n_items + 1) // 2 if name in HALF_BANK_SELECTORS else item_bank.n_items)


@pytest.mark.parametrize("name", sorted(SELECTOR_FACTORIES))
@given(params=BANKS, data=st.data(), seed=SEEDS)
@SEL_SETTINGS
def test_a_session_never_serves_an_excluded_item(name: str, params: np.ndarray, data: st.DataObject, seed: int) -> None:
  """With items excluded from the start, a session serves none of them and still ends with NoItemsAvailableError.

  A selector that drew its candidates without leaving out the excluded items, or that raised another error once only
  excluded items were left, fails; one that serves the whole bank must serve every item that is not excluded.
  """
  item_bank = ItemBank(params)
  excluded = data.draw(st.lists(st.integers(0, item_bank.n_items - 1), min_size=1, unique=True))

  picks = _session(SELECTOR_FACTORIES[name](item_bank.n_items), item_bank, data, seed=seed, excluded=excluded)

  if name not in HALF_BANK_SELECTORS:
    assert sorted(picks) == sorted(set(range(item_bank.n_items)) - set(excluded))


@given(
  params=item_params_matrix(min_items=4, max_items=16),
  data=st.data(),
  method=st.sampled_from(["item_info", "cluster_info", "weighted_info"]),
  seed=SEEDS,
)
@SEL_SETTINGS
def test_cluster_sessions_serve_the_whole_bank_for_any_labels(
  params: np.ndarray, data: st.DataObject, method: str, seed: int
) -> None:
  """Under every method, with labels that can skip values and leave clusters empty, a session serves the whole bank.

  A method that chose a label with no items, or a cluster whose items were all administered, would raise on an empty
  candidate set or return an administered item, and one that returned a numpy integer would let it into the session.
  """
  item_bank = ItemBank(params)
  n_clusters = data.draw(st.integers(min_value=1, max_value=item_bank.n_items // 2))
  clusters = data.draw(st.lists(st.integers(0, n_clusters - 1), min_size=item_bank.n_items, max_size=item_bank.n_items))

  picks = _session(ClusterSelector(clusters=clusters, method=method), item_bank, data, seed=seed)

  assert sorted(picks) == list(range(item_bank.n_items))
