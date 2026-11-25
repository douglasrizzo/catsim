"""Tests for catsim.Session, the immutable record of a CAT session (ADR-0006), and the checks it makes (ADR-0009)."""

from __future__ import annotations

import json
import math
import pickle
import re

import numpy as np
import pytest

from catsim import Estimate, Session, StopReason

INITIAL = Estimate(0.0, math.inf, 0)


@pytest.mark.parametrize(
  ("items", "responses", "thetas", "message"),
  [
    pytest.param(
      (0, 1),
      (True,),
      (0.0, 0.5),
      "responses has 1 entries and administered_items 2; there must be one response per administered item",
      id="item-without-response",
    ),
    pytest.param(
      (0,),
      (True,),
      (0.0,),
      "estimates has 1 entries and responses 1; a session holds the initial estimate and one estimate per response",
      id="missing-estimate",
    ),
    pytest.param(
      (0,),
      (True,),
      (0.0, 0.5, 0.7),
      "estimates has 3 entries and responses 1",
      id="extra-estimate",
    ),
    pytest.param((), (), (), "estimates has 0 entries and responses 0", id="no-initial-estimate"),
  ],
)
def test_a_session_rejects_a_history_that_does_not_line_up(
  items: tuple[int, ...], responses: tuple[bool, ...], thetas: tuple[float, ...], message: str
) -> None:
  """A session holds one response per item and one estimate more than responses.

  Fails if such a session is accepted: its current estimate, the last entry of the history, would not be the estimate
  after its last response (or, with an empty history, would not exist).
  """
  with pytest.raises(ValueError, match=re.escape(message)):
    Session(
      administered_items=items,
      responses=responses,
      estimates=tuple(Estimate(theta, math.inf) for theta in thetas),
    )


def test_a_session_stores_integer_and_numpy_responses_as_bools() -> None:
  """Answers stored as 0 and 1, or read back as NumPy Booleans, are accepted and held as Python bools.

  Fails if 1 and 0 are rejected (ADR-0009 accepts them), or if any is kept as given: the responses would mix types,
  ``json.dumps`` of a NumPy Boolean, the way an application stores a session, would raise, and a session rebuilt from
  storage would not equal the one stored.
  """
  session = Session(
    administered_items=(0, 1, 2),
    responses=(1, np.False_, np.int64(1)),  # ty: ignore[invalid-argument-type]
    estimates=(INITIAL, Estimate(0.5, 1.0), Estimate(0.3, 0.8), Estimate(0.6, 0.7)),
  )

  assert [type(response) for response in session.responses] == [bool, bool, bool]
  assert json.dumps(session.responses) == "[true, false, true]"


@pytest.mark.parametrize(
  ("response", "error", "message"),
  [
    pytest.param(1.0, TypeError, "responses[1] is 1.0; responses must be bools or the integers 0 and 1", id="float"),
    pytest.param("no", TypeError, "responses[1] is 'no'; responses must be bools or the integers 0 and 1", id="str"),
    pytest.param(-1, ValueError, "responses[1] is -1; an integer response must be 0 or 1", id="minus-one"),
  ],
)
def test_a_session_rejects_a_response_that_is_not_a_boolean_or_0_or_1(
  response: object, error: type[Exception], message: str
) -> None:
  """bool("no") and bool(-1) are True, so a session that converted them would hold answers never given."""
  with pytest.raises(error, match=re.escape(message)):
    Session(
      administered_items=(0, 1),
      responses=(True, response),  # ty: ignore[invalid-argument-type]
      estimates=(INITIAL, Estimate(0.5, 1.0), Estimate(0.3, 0.8)),
    )


@pytest.mark.parametrize(
  ("estimates", "message"),
  [
    pytest.param((0.0,), "estimates[0] is 0.0; a session's estimates must be Estimate objects", id="bare-theta"),
    pytest.param(({"theta": 0.0, "se": math.inf},), "estimates[0] is {'theta'", id="mapping"),
    pytest.param((INITIAL, 0.5), "estimates[1] is 0.5", id="one-bare-theta-among-estimates"),
  ],
)
def test_a_session_rejects_estimates_that_are_not_estimate_objects(estimates: tuple[object, ...], message: str) -> None:
  """A session restored with bare thetas or mappings is rejected when it is built, at the entry that is wrong.

  Fails if such a session is accepted: its theta history, its current SE and the engine's next step would fail on it
  later with an AttributeError.
  """
  n_responses = len(estimates) - 1

  with pytest.raises(TypeError, match=re.escape(message)):
    Session(
      administered_items=tuple(range(n_responses)),
      responses=(True,) * n_responses,
      estimates=estimates,  # ty: ignore[invalid-argument-type]
    )


def test_the_current_estimate_is_the_last_one() -> None:
  """current_theta and current_se read the last estimate, and theta_history every estimate in order.

  Fails if they read the initial estimate, another field, or drop the initial estimate from the history.
  """
  session = Session(
    administered_items=(2, 5),
    responses=(True, False),
    estimates=(INITIAL, Estimate(0.6, 1.1, 14), Estimate(0.4, 0.8, 12)),
  )

  assert (session.theta_history, session.current_theta, session.current_se) == ((0.0, 0.6, 0.4), 0.4, 0.8)


def test_a_stored_stop_reason_is_read_back_as_the_enum() -> None:
  """A stop reason stored as its string is the StopReason member again; an unknown one is rejected.

  Fails if the string is kept, so that ``stop_reason is StopReason.STOPPER`` is False for a restored session, or if a
  reason catsim never sets is accepted.
  """
  stored = json.loads(json.dumps({"stop_reason": StopReason.STOPPER, "unknown": "paused"}))

  restored = Session(estimates=(INITIAL,), stop_reason=stored["stop_reason"])

  assert restored.stop_reason is StopReason.STOPPER
  with pytest.raises(
    ValueError, match=re.escape("stop_reason is 'paused'; a stop reason must be None or one of 'stopper', 'no_items'")
  ):
    Session(estimates=(INITIAL,), stop_reason=stored["unknown"])


@pytest.mark.parametrize(
  ("excluded", "message"),
  [
    pytest.param(
      (4, 4),
      "excluded_items[1] is 4; item indices must not repeat, and excluded_items[0] is 4 too",
      id="repeated",
    ),
    pytest.param(
      (2, 1),
      "excluded_items[1] is 1; an item cannot be both administered and excluded, and administered_items[0] is 1 too",
      id="administered",
    ),
  ],
)
def test_a_session_rejects_excluded_items_it_cannot_hold(excluded: tuple[int, ...], message: str) -> None:
  """An excluded item twice, or an item both administered and excluded, is rejected.

  Fails if such a session is accepted: an item is given or declined once per session, so a record holding either is
  corrupt, and its proposal counts (administrations plus declines) would count an item twice for one examinee.
  """
  with pytest.raises(ValueError, match=re.escape(message)):
    Session(
      administered_items=(1,), responses=(True,), estimates=(INITIAL, Estimate(0.5, 1.0)), excluded_items=excluded
    )


def test_a_session_rejects_a_repeated_administered_item() -> None:
  """A session that gave an item twice would count its response twice in every estimate after it."""
  with pytest.raises(
    ValueError,
    match=re.escape("administered_items[1] is 3; item indices must not repeat, and administered_items[0] is 3 too"),
  ):
    Session(
      administered_items=(3, 3), responses=(True, True), estimates=(INITIAL, Estimate(0.5, 1.0), Estimate(0.9, 0.8))
    )


@pytest.mark.parametrize("field_name", ["administered_items", "excluded_items"])
@pytest.mark.parametrize(
  ("entry", "error", "rule"),
  [
    pytest.param(2.5, TypeError, "item indices must be integers", id="float"),
    pytest.param("3", TypeError, "item indices must be integers", id="str"),
    pytest.param(-1, ValueError, "item indices must not be negative", id="negative"),
  ],
)
def test_a_session_rejects_item_indices_that_are_not_items(
  field_name: str, entry: object, error: type[Exception], rule: str
) -> None:
  """An index that is not a non-negative integer fails when the session is built, naming the field and the entry.

  Fails if 2.5 is accepted (NumPy truncates it to item 2, so an excluded item could be given or an administered one
  given again), if "3" is accepted and the session breaks later inside a strategy, or if -1 is accepted and NumPy
  reads it as the bank's last item. The upper bound is checked by the engine, which knows the bank.
  """
  administered, excluded = ((entry,), ()) if field_name == "administered_items" else ((), (entry,))

  with pytest.raises(error, match=re.escape(f"{field_name}[0] is {entry!r}; {rule}")):
    Session(
      administered_items=administered,  # ty: ignore[invalid-argument-type]
      responses=(True,) * len(administered),
      estimates=(INITIAL,) * (len(administered) + 1),
      excluded_items=excluded,  # ty: ignore[invalid-argument-type]
    )


def test_a_session_stores_numpy_integer_indices_as_python_ints() -> None:
  """Item indices restored as NumPy integers, such as from a stored array, are stored as builtin ints.

  Fails if the session keeps the numpy.int64 entries: storing it again with json.dumps would raise TypeError.
  """
  session = Session(
    administered_items=(np.int64(4),),  # ty: ignore[invalid-argument-type]
    responses=(True,),
    estimates=(INITIAL, Estimate(0.5, 1.0)),
    excluded_items=(np.int64(7),),  # ty: ignore[invalid-argument-type]
  )

  assert json.dumps([session.administered_items, session.excluded_items]) == "[[4], [7]]"
  assert [type(item) for item in session.administered_items + session.excluded_items] == [int, int]


def test_a_session_keeps_its_own_copy_of_the_sequences_it_is_given() -> None:
  """Lists passed to a session are copied into tuples, so changing them later leaves the session as it was.

  Fails if the session keeps the caller's lists: the record of a test would change after the fact (ADR-0006).
  """
  items, responses, excluded = [6], [True], [2]
  session = Session(
    administered_items=items,  # ty: ignore[invalid-argument-type]
    responses=responses,  # ty: ignore[invalid-argument-type]
    estimates=(INITIAL, Estimate(0.5, 1.0)),
    excluded_items=excluded,  # ty: ignore[invalid-argument-type]
  )

  items.append(3)
  responses.append(False)
  excluded.append(9)

  assert (session.administered_items, session.responses, session.excluded_items) == ((6,), (True,), (2,))


def test_a_session_is_a_value_that_hashes_and_pickles() -> None:
  """Equal sessions hash alike and a pickled session comes back equal, so sessions can be stored, deduplicated and sent.

  Fails if a field held a list or a NumPy array (``hash`` would raise TypeError), if sessions compared by identity (two
  records of the same test would differ), or if the class could not be pickled to a worker process.
  """
  first = Session(administered_items=(3,), responses=(True,), estimates=(INITIAL, Estimate(0.5, 1.0, 4)))
  same = Session(administered_items=[3], responses=[1], estimates=[INITIAL, Estimate(0.5, 1.0, 4)])  # ty: ignore[invalid-argument-type]
  later = Session(
    administered_items=(3, 4), responses=(True, False), estimates=(INITIAL, Estimate(0.5, 1.0, 4), Estimate(0.2, 0.9))
  )

  assert len({first, same, later}) == 2
  assert pickle.loads(pickle.dumps(later)) == later
