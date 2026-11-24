"""Property-based tests for ability estimation."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from catsim import irt
from catsim.estimation import NumericalSearchEstimator
from catsim.irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from catsim.item_bank import ItemBank
from tests.contexts import make_context
from tests.hypothesis_strategies import item_params_matrix

EST_SETTINGS = settings(max_examples=15, deadline=None)


@given(
  params=item_params_matrix(min_items=8, max_items=18),
  k=st.integers(min_value=4, max_value=10),
  data=st.data(),
)
@EST_SETTINGS
def test_estimate_is_a_likelihood_maximum_on_the_ability_scale(
  params: np.ndarray,
  k: int,
  data,
) -> None:
  """A mixed pattern's estimate lies in [-6, 6], and no point 1e-3 either side beats it by 1e-4.

  SciPy's brent and golden searches, given (-6, 6) as a starting bracket, left the scale on
  guessing items. Next to a bound, where a mixed pattern's likelihood can still be rising, the
  clipped neighbour can be more likely by the slope times the 1e-6 tol, well under the 1e-4
  allowed.
  """
  n = params.shape[0]
  assume(k <= n)
  administered_items = list(range(k))
  response_vector = data.draw(st.lists(st.booleans(), min_size=k, max_size=k))
  assume(len(set(response_vector)) > 1)
  bank = ItemBank(params, validate=True)
  items = bank.get_items(administered_items)
  estimator = NumericalSearchEstimator()
  theta = estimator.estimate(make_context(bank, administered_items, response_vector, theta=0.0))
  neighbours = np.clip([theta - 1e-3, theta + 1e-3], THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)

  assert THETA_MIN_EXTENDED <= theta <= THETA_MAX_EXTENDED
  assert max(irt.log_likelihood(x, response_vector, items) for x in neighbours) <= (
    irt.log_likelihood(theta, response_vector, items) + 1e-4
  )
