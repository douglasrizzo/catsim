"""Simulated CAT runs on an engine, and their aggregate results."""

from __future__ import annotations

import enum
import functools
import itertools
import math
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, cast, override

import numpy

from . import metrics
from .checks import _named
from .engine import CatEngine
from .item_bank import ItemBank
from .metrics import Metric
from .session import Session
from .workers import _payload, _run_on_workers

if TYPE_CHECKING:
  from collections.abc import Callable, Iterable, Iterator, Sequence

  import numpy.typing as npt

  from .typing import FloatArray, IntArray


class _Stream(enum.IntEnum):
  """The last entry of an examinee's spawn key, which says which of its two random streams it is (ADR-0013)."""

  PATTERN = 0  # the examinee's answers to every item, drawn before its test
  SESSION = 1  # the generator of its session's steps


def simulate(  # noqa: PLR0913 - the engine, two ways to give examinees, the seed, the split and progress (ADR-0015)
  engine: CatEngine,
  *,
  true_thetas: npt.ArrayLike | None = None,
  responses: npt.ArrayLike | None = None,
  rng: int | numpy.random.Generator | None = None,
  parts: int = 1,
  workers: int = 1,
  progress: Callable[[int, int], object] | None = None,
) -> SimulationResult:
  r"""Run one CAT session per examinee on an engine, in parts that can run on several processes.

  Each examinee has an answer to every item of the bank, possibly missing for a recorded examinee,
  and its session takes the answer to each item it is given from them (ADR-0014). The examinees are
  given in one of three ways:

  - ``true_thetas`` alone: generated examinees. Before its test, examinee ``i`` draws one uniform
    number :math:`u_j` per item of the bank, and its answer to item :math:`j` is correct when
    :math:`u_j < P_j(\theta_i)`, the item's probability of a correct answer at the true ability
    :math:`\theta_i` (``true_thetas[i]``).
  - ``responses`` alone: recorded examinees, whose true abilities are unknown. Row ``i`` holds
    examinee ``i``'s answers. The items whose answer is missing are excluded from the start of the
    session (ADR-0011), so a recorded examinee is given only items they answered, and one who runs
    out of them ends with ``StopReason.NO_ITEMS``.
  - Both: recorded examinees whose true abilities are known, as when a study draws its own response
    matrix with a response model of its choice.

  Every random draw comes from ``rng`` (ADR-0013). The run takes one root entropy from it, and
  examinee ``i``, at position ``i`` of the run, draws from two streams of its own:
  ``SeedSequence(entropy, spawn_key=(i, 0))`` for its answers to every item, and
  ``SeedSequence(entropy, spawn_key=(i, 1))`` for its session, the generator of its ``start``,
  ``select`` and ``answer`` steps, from which the initializer, the selector and the exposure control
  draw. A generated examinee therefore gives the same answer to an item whatever the engine and
  whenever the item comes, so two engines run on the same seed and examinees are compared on common
  random numbers.

  The examinees are split, in order, into ``parts`` contiguous parts of near-equal size, and each
  part runs its sessions one after another (ADR-0015). Each session's selections are capped
  against the items' exposure rates: each item's administrations over the sessions before it in its
  part (ADR-0012), so the first session of a part sees no rates. This exposure feedback is the only
  way one session depends on another, and the parts are part of the run's configuration: with an
  ``r_max`` cap, a session sees only the exposure of its own part, so ``parts`` changes the result,
  and ``parts=1``, the default, is the sequential run. Without a cap, an examinee's session depends
  on neither the other examinees nor ``parts``.

  The parts run on up to ``workers`` processes, which never change the result: it depends only on
  the engine, the examinees, the seed and ``parts``. With more than one part and more than one
  worker, the parts run on processes of Python's standard library, spawned on every platform. Each
  receives the engine pickled and imports every class and function it refers to by its module and
  name, so a strategy defined in a notebook or a script's main module, a lambda (such as a custom
  ``log_prior``) or a function defined inside another function raises before any part runs: define
  it in a module and import it from there. A script that runs on workers calls ``simulate`` under
  ``if __name__ == "__main__":``, as every process pool of Python requires.

  The :doc:`/running-simulations` guide runs a study and reads its result.

  Parameters
  ----------
  engine : CatEngine
      The engine whose item bank and strategies every session uses.
  true_thetas : array-like or None, optional
      The examinees' true abilities: a non-empty, one-dimensional sequence of finite numbers. To
      simulate ``n`` examinees from a population, draw their abilities, e.g.
      ``numpy.random.default_rng(1).normal(0, 1, n)``. Default is None, for recorded examinees whose
      abilities are unknown.
  responses : array-like or None, optional
      The recorded examinees' answers: a matrix with one row per examinee and one column per item of
      the bank, of Booleans, of the integers 0 and 1, or of the floats 0.0 and 1.0 with NaN where an
      answer is missing. Default is None, for generated examinees.
  rng : int, numpy.random.Generator or None, optional
      The source of the run's root entropy. None, the default, takes fresh entropy, so every call
      gives other tests. A non-negative int is the entropy itself, so the same seed gives the same
      result. From a generator, one 128-bit number is drawn as the entropy, which advances it, so
      passing the same generator to another run gives other results; a call whose inputs fail
      their checks draws nothing from it.
  parts : int, optional
      The number of parts the examinees are split into, from 1 to the number of examinees: each
      part holds the next examinees in order, and the first ``n % parts`` of the parts of ``n``
      examinees have one examinee more than the others, the sizes of ``numpy.array_split``. Default
      is 1, the sequential run.
  workers : int, optional
      The most processes the parts run on at once, a positive integer. With 1, the default, or with
      one part, the calling process runs the parts itself, one after another; otherwise
      ``min(workers, parts)`` worker processes run them.
  progress : callable or None, optional
      Called in the calling process as ``progress(finished, total)`` after each session ends, with
      the number of sessions finished and the number of examinees, such as
      ``lambda finished, total: bar.update(1)`` for a progress bar ``bar``. On workers, the sessions
      end in the order the processes run them, and each call still counts one more session.
      Default is None.

  Returns
  -------
  SimulationResult
      The sessions, in examinee order, with the true abilities (None for recorded examinees whose
      abilities are unknown), the run's exposure and proposal counts, summed over its parts, the
      number of parts, and the run's duration: the wall-clock time in the calling process from the
      start of the first part to the end of the last. On workers it includes starting them and
      sending them the engine, so it depends on ``workers`` and on the machine, which the sessions
      do not.

  Raises
  ------
  TypeError
      If ``engine`` is not a :class:`~catsim.CatEngine`; neither ``true_thetas`` nor ``responses``
      is given; ``true_thetas`` is a single number, holds values that are not real numbers or has
      more than one dimension; ``responses`` is not two-dimensional or holds values that are neither
      Booleans nor real numbers; ``rng`` is neither None, an integer nor a generator; ``parts`` or
      ``workers`` is not an integer; or ``progress`` is neither None nor callable.
  ValueError
      If ``true_thetas`` or ``responses`` is empty; a true ability is not finite; ``responses`` does
      not have one column per item of the bank, or holds a response other than 0, 1 and NaN; the two
      inputs have different numbers of examinees; ``rng`` is a negative integer; ``parts`` or
      ``workers`` is less than 1; ``parts`` exceeds the number of examinees; or, when the parts run
      on workers, the engine refers to a class or function the workers cannot import, or cannot be
      pickled.

  Examples
  --------
  Generated examinees, drawn from a standard normal population:

  >>> import numpy
  >>> from catsim import CatEngine, ItemBank, simulate
  >>> from catsim.estimation import EAPEstimator
  >>> from catsim.initialization import FixedPointInitializer
  >>> from catsim.selection import MaxInfoSelector
  >>> from catsim.stopping import MaxLengthStopper
  >>> bank = ItemBank.generate(50, rng=0)
  >>> engine = CatEngine(bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), MaxLengthStopper(10))
  >>> thetas = numpy.random.default_rng(1).normal(0, 1, 100)
  >>> result = simulate(engine, true_thetas=thetas, rng=0)
  >>> len(result.sessions), result.examinees()["test_length"].tolist() == [10] * 100
  (100, True)

  Without an exposure cap, the run's four parts give the sequential run's sessions, and two
  workers give the same sessions as one:

  >>> in_parts = simulate(engine, true_thetas=thetas, rng=0, parts=4, workers=2)
  >>> in_parts.parts, in_parts.sessions == result.sessions
  (4, True)

  Two recorded examinees: the first answered every item correctly, the second only items 3, 8 and
  15, so the second runs out of answered items after three:

  >>> responses = numpy.full((2, bank.n_items), numpy.nan)
  >>> responses[0] = 1.0
  >>> responses[1, [3, 8, 15]] = [1.0, 0.0, 1.0]
  >>> result = simulate(engine, responses=responses, rng=0)
  >>> [(len(session.administered_items), str(session.stop_reason)) for session in result.sessions]
  [(10, 'stopper'), (3, 'no_items')]
  >>> sorted(result.sessions[1].administered_items)
  [3, 8, 15]
  >>> result.rmse()  # doctest: +ELLIPSIS
  Traceback (most recent call last):
  ...
  ValueError: rmse needs the examinees' true abilities, and this result has none; ...
  """
  if not isinstance(engine, CatEngine):
    msg = f"engine is a {type(engine).__name__}; it must be a CatEngine"
    raise TypeError(msg)
  if progress is not None and not callable(progress):
    msg = f"progress is {progress!r}; it must be None or a callable, called as progress(finished, total)"
    raise TypeError(msg)
  parts = _positive("parts", parts)
  workers = _positive("workers", workers)
  thetas, matrix, n_examinees = _examinees(true_thetas, responses, engine.item_bank.names)
  if parts > n_examinees:
    msg = f"parts is {parts} and the run has {n_examinees} examinees; every part needs at least one examinee"
    raise ValueError(msg)
  processes = min(workers, parts)
  # The engine is checked, and pickled for the workers, before anything runs.
  payload = _payload(engine) if processes > 1 else None
  # Last, so that a call that fails its checks does not advance a generator passed as rng.
  entropy = _entropy(rng)

  split = _split(thetas, matrix, n_examinees, parts, entropy)
  done = None if progress is None else _counter(progress, n_examinees)
  start = time.perf_counter()
  if payload is None:
    results = [_run_part(engine, part, done) for part in split]
  else:
    results = _run_on_workers(_run_part, payload, split, workers=processes, done=done)
  duration = time.perf_counter() - start

  return SimulationResult(
    item_bank=engine.item_bank,
    sessions=tuple(session for result in results for session in result.sessions),
    proposal_counts=numpy.sum([result.proposal_counts for result in results], axis=0),
    duration=duration,
    true_thetas=thetas,
    parts=parts,
  )


def _positive(name: str, value: object) -> int:
  """Return ``parts`` or ``workers`` as an int, checking that it is a positive integer.

  Parameters
  ----------
  name : str
      The argument's name, for the error messages.
  value : object
      The value passed to :func:`simulate`.

  Returns
  -------
  int
      The value.

  Raises
  ------
  TypeError
      If the value is not an integer, or is a Boolean.
  ValueError
      If the value is less than 1.
  """
  # A bool is an int, but parts=True is a flag passed by mistake.
  if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, (int, numpy.integer)):
    msg = f"{name} is {value!r}; it must be a positive integer"
    raise TypeError(msg)
  if value < 1:
    msg = f"{name} is {value}; it must be a positive integer"
    raise ValueError(msg)
  return int(value)


def _examinees(
  true_thetas: npt.ArrayLike | None, responses: npt.ArrayLike | None, names: Sequence[str]
) -> tuple[FloatArray | None, FloatArray | None, int]:
  """Return a run's checked true abilities and recorded responses, either None when not given, and how many examinees.

  Parameters
  ----------
  true_thetas : array-like or None
      The true abilities passed to :func:`simulate`.
  responses : array-like or None
      The recorded responses passed to :func:`simulate`.
  names : Sequence[str]
      The names of the items of the engine's bank.

  Returns
  -------
  tuple[numpy.ndarray or None, numpy.ndarray or None, int]
      The true abilities as a float array, the responses as a float matrix with NaN where an answer
      is missing, and the number of examinees.

  Raises
  ------
  TypeError
      If neither input is given.
  ValueError
      If both are given, with different numbers of examinees.
  """
  matrix = None if responses is None else _response_matrix(responses, names)
  if true_thetas is None:
    if matrix is None:
      msg = "simulate needs examinees: true_thetas for generated ones, responses for recorded ones, or both"
      raise TypeError(msg)
    return None, matrix, len(matrix)
  thetas = _true_thetas(true_thetas)
  if matrix is not None and len(thetas) != len(matrix):
    msg = (
      f"true_thetas has {len(thetas)} entries and responses {len(matrix)} rows; with both, there must be one "
      "true ability per row of responses"
    )
    raise ValueError(msg)
  return thetas, matrix, len(thetas)


def _true_thetas(true_thetas: npt.ArrayLike) -> FloatArray:
  """Return the true abilities as a float array, checking there is one finite number per examinee.

  Parameters
  ----------
  true_thetas : array-like
      The true abilities passed to :func:`simulate`.

  Returns
  -------
  numpy.ndarray
      The abilities as a one-dimensional float array.

  Raises
  ------
  TypeError
      If ``true_thetas`` is a single number, holds values that are not real numbers or has more than
      one dimension.
  ValueError
      If ``true_thetas`` is empty or holds a value that is not finite.
  """
  thetas = numpy.asarray(true_thetas)
  if thetas.ndim == 0:
    msg = (
      f"true_thetas must hold one true ability per examinee, not a single number ({true_thetas!r}); "
      "to simulate n examinees, pass their abilities, e.g. rng.normal(0, 1, n)"
    )
    raise TypeError(msg)
  # Booleans would be read as abilities 0 and 1, and None as NaN.
  if thetas.size and thetas.dtype.kind not in "iuf":
    msg = f"true_thetas holds {thetas.dtype} values; true abilities must be real numbers"
    raise TypeError(msg)
  if thetas.ndim != 1:
    msg = f"true_thetas must be one-dimensional, one true ability per examinee, got shape {thetas.shape}"
    raise TypeError(msg)
  if thetas.size == 0:
    msg = "true_thetas cannot be empty"
    raise ValueError(msg)
  thetas = thetas.astype(float)
  # The probability of a correct answer at NaN is NaN, so every generated answer would be wrong.
  if not numpy.isfinite(thetas).all():
    position = int(numpy.flatnonzero(~numpy.isfinite(thetas))[0])
    msg = f"true_thetas[{position}] is {thetas[position].item()!r}; true abilities must be finite numbers"
    raise ValueError(msg)
  return thetas


def _response_matrix(responses: npt.ArrayLike, names: Sequence[str]) -> FloatArray:
  """Return recorded responses as a float matrix of 1.0 (correct), 0.0 (incorrect) and NaN (missing), checking them.

  Parameters
  ----------
  responses : array-like
      The recorded responses passed to :func:`simulate`.
  names : Sequence[str]
      The names of the items of the engine's bank, one per column, which the messages give.

  Returns
  -------
  numpy.ndarray
      A copy of the responses, one row per examinee and one column per item.

  Raises
  ------
  TypeError
      If the responses are not two-dimensional, or hold values that are neither Booleans nor real
      numbers.
  ValueError
      If there is no row, not one column per item, or a response other than 0, 1 and, among floats,
      NaN.
  """
  matrix = numpy.asarray(responses)
  # Before the shape check: an empty list has one dimension, and is empty like true_thetas=[].
  if matrix.ndim in {1, 2} and len(matrix) == 0:
    msg = "responses cannot be empty"
    raise ValueError(msg)
  if matrix.ndim != 2:  # noqa: PLR2004 - a matrix
    msg = (
      "responses must be two-dimensional, one row per examinee and one column per item of the bank, got shape "
      f"{matrix.shape}"
    )
    raise TypeError(msg)
  if matrix.shape[1] != len(names):
    msg = f"responses has shape {matrix.shape}; there must be one column per item of the bank ({len(names)})"
    raise ValueError(msg)
  kind = matrix.dtype.kind
  if kind == "b":
    return matrix.astype(float)
  # Strings and objects are no responses: None would mark a missing answer that NaN marks.
  if kind not in "iuf":
    msg = (
      f"responses holds {matrix.dtype} values; responses must be bools, the integers 0 and 1, or the floats 0.0 "
      "and 1.0 with NaN for a missing response"
    )
    raise TypeError(msg)
  valid = (matrix == 0) | (matrix == 1)
  if kind == "f":
    valid |= numpy.isnan(matrix)
  if not valid.all():
    row, column = (int(index) for index in numpy.argwhere(~valid)[0])
    rule = (
      "an integer response must be 0 or 1" if kind in "iu" else "a response must be 0 or 1, or NaN where it is missing"
    )
    msg = f"responses[{row}, {column}] is {matrix[row, column].item()!r}, for item {_named(column, names)}; {rule}"
    raise ValueError(msg)
  return matrix.astype(float)


def _entropy(rng: object) -> int:
  """Return a run's root entropy: the seed itself, a number drawn once from a generator, or fresh entropy (ADR-0013).

  Parameters
  ----------
  rng : object
      The ``rng`` passed to :func:`simulate`.

  Returns
  -------
  int
      The root entropy of the examinees' seed sequences.

  Raises
  ------
  TypeError
      If ``rng`` is neither None, an integer nor a generator.
  ValueError
      If ``rng`` is a negative integer.
  """
  if rng is None:
    # The size of SeedSequence's own fresh entropy.
    return secrets.randbits(128)
  if isinstance(rng, numpy.random.Generator):
    return int.from_bytes(rng.bytes(16), "little")
  # A bool is an int, but True passed as a seed is a flag passed by mistake.
  if isinstance(rng, (bool, numpy.bool_)) or not isinstance(rng, (int, numpy.integer)):
    msg = f"rng is {rng!r}; it must be None, an integer seed or a numpy.random.Generator"
    raise TypeError(msg)
  if rng < 0:
    msg = f"rng is {rng}; a seed must not be negative"
    raise ValueError(msg)
  return int(rng)


def _stream(entropy: int, examinee: int, stream: _Stream) -> numpy.random.Generator:
  """Return the generator of one of an examinee's two random streams (ADR-0013).

  Parameters
  ----------
  entropy : int
      The run's root entropy.
  examinee : int
      The examinee's position in the run.
  stream : _Stream
      Which of the examinee's streams.

  Returns
  -------
  numpy.random.Generator
      A generator seeded by ``SeedSequence(entropy, spawn_key=(examinee, stream))``.
  """
  return numpy.random.default_rng(numpy.random.SeedSequence(entropy, spawn_key=(examinee, stream)))


class _Part(NamedTuple):
  """A contiguous part of a run's examinees, with all a worker needs, besides the engine, to run its sessions."""

  first: int  # the position in the run of the part's first examinee, which keys the streams (ADR-0013)
  true_thetas: FloatArray | None  # the part's true abilities, or None
  responses: FloatArray | None  # the part's recorded rows, or None
  entropy: int  # the run's root entropy


class _PartResult(NamedTuple):
  """The sessions of a part, in examinee order, and the part's exposure and proposal counts."""

  sessions: list[Session]
  exposure_counts: IntArray
  proposal_counts: IntArray


def _split(
  true_thetas: FloatArray | None, responses: FloatArray | None, n_examinees: int, parts: int, entropy: int
) -> list[_Part]:
  """Split a run's examinees, in order, into contiguous parts with the sizes of ``numpy.array_split``.

  Parameters
  ----------
  true_thetas : numpy.ndarray or None
      The examinees' true abilities, or None.
  responses : numpy.ndarray or None
      The recorded responses, or None.
  n_examinees : int
      The number of examinees.
  parts : int
      The number of parts, at most the number of examinees.
  entropy : int
      The run's root entropy.

  Returns
  -------
  list of _Part
      The parts, in examinee order.
  """
  split = []
  for positions in numpy.array_split(numpy.arange(n_examinees), parts):
    first, end = int(positions[0]), int(positions[-1]) + 1
    split.append(
      _Part(
        first=first,
        true_thetas=None if true_thetas is None else true_thetas[first:end],
        responses=None if responses is None else responses[first:end],
        entropy=entropy,
      )
    )
  return split


def _counter(progress: Callable[[int, int], object], total: int) -> Callable[[], None]:
  """Return a function that tells ``progress`` that one more of ``total`` sessions has finished.

  Parameters
  ----------
  progress : callable
      The ``progress`` passed to :func:`simulate`.
  total : int
      The number of examinees.

  Returns
  -------
  callable
      A function of no arguments, called once after each session ends.
  """
  finished = itertools.count(1)

  def done() -> None:
    progress(next(finished), total)

  return done


def _run_part(engine: CatEngine, part: _Part, done: Callable[[], object] | None) -> _PartResult:
  """Run a part's sessions one after another, with exposure counts of the part's own (ADR-0015).

  Parameters
  ----------
  engine : CatEngine
      The engine.
  part : _Part
      The part.
  done : callable or None
      Called after each session ends, or None.

  Returns
  -------
  _PartResult
      The part's sessions, in examinee order, and its exposure and proposal counts.
  """
  exposure_counts = numpy.zeros(engine.item_bank.n_items, dtype=int)
  proposal_counts = numpy.zeros(engine.item_bank.n_items, dtype=int)
  sessions: list[Session] = []
  for before, row in enumerate(_rows(engine.item_bank, part)):
    # Each item's administrations over the sessions of the part started before this one,
    # so the part's first session sees none (ADR-0012). Only this session changes the counts until
    # it ends, and it is never offered an item again once given it, so its rates are fixed when it
    # starts.
    rates = exposure_counts / before if before else None
    if rates is not None:
      rates.flags.writeable = False
    session, declined = _run_session(engine, row, _stream(part.entropy, part.first + before, _Stream.SESSION), rates)
    for item in session.administered_items:
      exposure_counts[item] += 1
    for item in (*session.administered_items, *declined):
      proposal_counts[item] += 1
    sessions.append(session)
    if done is not None:
      done()
  return _PartResult(sessions, exposure_counts, proposal_counts)


def _rows(item_bank: ItemBank, part: _Part) -> Iterator[FloatArray]:
  """Yield each examinee's answers to every item, in the part's order: the recorded row, or one drawn before the test.

  A generated examinee draws one uniform number per item from its pattern stream, keyed by its
  position in the run, and its answer to an item is correct when the number falls below the item's
  probability at its true ability (ADR-0014).

  Parameters
  ----------
  item_bank : ItemBank
      The engine's item bank.
  part : _Part
      The part.

  Yields
  ------
  numpy.ndarray
      The examinee's row: 1.0 for a correct answer, 0.0 for an incorrect one and NaN for a missing
      one.
  """
  if part.responses is not None:
    yield from part.responses
  elif part.true_thetas is not None:
    for i, true_theta in enumerate(part.true_thetas.tolist(), start=part.first):
      draws = _stream(part.entropy, i, _Stream.PATTERN).random(item_bank.n_items)
      yield (draws < item_bank.icc(true_theta)).astype(float)


def _run_session(
  engine: CatEngine, row: FloatArray, rng: numpy.random.Generator, exposure_rates: FloatArray | None
) -> tuple[Session, list[int]]:
  """Run one session to its end, answering from ``row``, and return it with the items the exposure control declined.

  The items whose answer is missing are excluded from the start. :func:`simulate` builds every
  input of the engine's steps itself, so it takes the engine's unchecked path (ADR-0009).

  Parameters
  ----------
  engine : CatEngine
      The engine.
  row : numpy.ndarray
      The examinee's answers to every item: 1.0 for correct, 0.0 for incorrect and NaN for missing.
  rng : numpy.random.Generator
      The generator of the session's steps.
  exposure_rates : numpy.ndarray or None
      The exposure rates the session's selections are capped against.

  Returns
  -------
  tuple[Session, list[int]]
      The finished session, and the items the exposure control declined in it, in order.
  """
  session = engine._start(rng, tuple(numpy.flatnonzero(numpy.isnan(row)).tolist()))  # noqa: SLF001
  declined: list[int] = []
  while True:
    selection = engine._select(session, rng, exposure_rates)  # noqa: SLF001
    session = selection.session
    declined.extend(selection.declined)
    if selection.item is None:
      return session, declined
    # The row holds 1.0 or 0.0 for every item the session may be given; its NaNs are excluded.
    session = engine._answer(session, selection.item, bool(row[selection.item]), rng)  # noqa: SLF001


_N_DECILES = 10


def _deciles(true_thetas: FloatArray) -> IntArray:
  """Return each examinee's decile of true ability, from 0 for the lowest abilities to 9.

  Parameters
  ----------
  true_thetas : numpy.ndarray
      The examinees' true abilities.

  Returns
  -------
  numpy.ndarray
      Each examinee's decile. :class:`SimulationResult` states the definition.
  """
  # The cuts are at exactly k / 10: catR's seq(0, 1, 0.1) puts 0.3, 0.6 and 0.7 a rounding error above.
  cuts = numpy.quantile(true_thetas, numpy.arange(_N_DECILES + 1) / _N_DECILES)
  # R's findInterval: the number of cuts at or below each ability, and the highest abilities join the top decile.
  return numpy.minimum(numpy.searchsorted(cuts, true_thetas, side="right"), _N_DECILES) - 1


def _read_only[Array: numpy.ndarray](array: Array) -> Array:
  """Return ``array`` after making it read-only."""
  array.flags.writeable = False
  return array


def _finished_sessions(sessions: Iterable[object]) -> tuple[Session, ...]:
  """Return a result's sessions as a tuple, checking that there is at least one and that each has ended.

  Parameters
  ----------
  sessions : Iterable[object]
      The sessions passed to :class:`SimulationResult`.

  Returns
  -------
  tuple[Session, ...]
      The sessions.

  Raises
  ------
  TypeError
      If an entry is not a :class:`~catsim.Session`.
  ValueError
      If there is no session, or one has not ended.
  """
  checked = tuple(sessions)
  if not checked:
    msg = "sessions is empty; a result holds at least one session"
    raise ValueError(msg)
  for position, session in enumerate(checked):
    if not isinstance(session, Session):
      msg = f"sessions[{position}] is a {type(session).__name__}; a result's sessions must be Session objects"
      raise TypeError(msg)
    if session.stop_reason is None:
      msg = f"sessions[{position}] has not ended, its stop_reason is None; a result holds finished sessions"
      raise ValueError(msg)
  return cast("tuple[Session, ...]", checked)


def _counts(sessions: Sequence[Session], n_items: int) -> IntArray:
  """Return how many of the sessions administered each item.

  Parameters
  ----------
  sessions : Sequence[Session]
      The sessions.
  n_items : int
      The number of items in the bank, at least the largest administered index plus one.

  Returns
  -------
  numpy.ndarray
      One count per item.
  """
  items = numpy.fromiter(itertools.chain.from_iterable(session.administered_items for session in sessions), dtype=int)
  return numpy.bincount(items, minlength=n_items)


def _check_items(sessions: Sequence[Session], n_items: int) -> None:
  """Check that every administered item is in the bank.

  Parameters
  ----------
  sessions : Sequence[Session]
      The sessions.
  n_items : int
      The number of items in the bank.

  Raises
  ------
  ValueError
      If a session administered an index the bank does not have.
  """
  for position, session in enumerate(sessions):
    for step, item in enumerate(session.administered_items):
      if item >= n_items:
        msg = (
          f"sessions[{position}].administered_items[{step}] is {item}; the item bank has {n_items} items, 0 to "
          f"{n_items - 1}"
        )
        raise ValueError(msg)


def _proposal_counts(
  proposal_counts: npt.ArrayLike, exposure_counts: IntArray, n_sessions: int, names: Sequence[str]
) -> IntArray:
  """Return a result's proposal counts as a read-only integer array, checking them against its administrations.

  Parameters
  ----------
  proposal_counts : array-like
      The proposal counts passed to :class:`SimulationResult`.
  exposure_counts : numpy.ndarray
      Each item's administrations.
  n_sessions : int
      The number of sessions.
  names : Sequence[str]
      The names of the bank's items, which the message gives.

  Returns
  -------
  numpy.ndarray
      A read-only copy of the counts.

  Raises
  ------
  TypeError
      If the counts are not integers.
  ValueError
      If there is not one count per item, or a count is below the item's administrations or above
      the number of sessions.
  """
  counts = numpy.asarray(proposal_counts)
  if counts.dtype.kind not in "iu":
    msg = f"proposal_counts holds {counts.dtype} values; counts must be integers"
    raise TypeError(msg)
  if counts.shape != exposure_counts.shape:
    msg = (
      f"proposal_counts has shape {counts.shape}; there must be one count per item of the bank ({len(exposure_counts)})"
    )
    raise ValueError(msg)
  # A session proposes an item at most once: the exposure control declines it or the session administers it.
  outside = (counts < exposure_counts) | (counts > n_sessions)
  if outside.any():
    item = int(numpy.flatnonzero(outside)[0])
    msg = (
      f"proposal_counts[{item}] is {counts[item].item()}, for item {_named(item, names)}; an item's proposals are "
      f"at least its {exposure_counts[item]} administrations and at most the {n_sessions} sessions"
    )
    raise ValueError(msg)
  return _read_only(numpy.array(counts, dtype=int))


def _result_true_thetas(true_thetas: npt.ArrayLike, n_sessions: int) -> FloatArray:
  """Return a result's true abilities as a read-only float array, checking that there is one finite number per session.

  Parameters
  ----------
  true_thetas : array-like
      The true abilities passed to :class:`SimulationResult`.
  n_sessions : int
      The number of sessions.

  Returns
  -------
  numpy.ndarray
      A read-only copy of the true abilities.

  Raises
  ------
  ValueError
      If there is not one true ability per session, or one is not finite.
  """
  thetas = numpy.array(true_thetas, dtype=float)
  if thetas.shape != (n_sessions,):
    msg = f"true_thetas has shape {thetas.shape}; there must be one true ability per session ({n_sessions})"
    raise ValueError(msg)
  # The true abilities are known for every session or for none (ADR-0016): NaN does not mark an unknown one.
  if not numpy.isfinite(thetas).all():
    position = int(numpy.flatnonzero(~numpy.isfinite(thetas))[0])
    msg = (
      f"true_thetas[{position}] is {thetas[position].item()!r}; true abilities must be finite, and true_thetas None "
      "when they are unknown"
    )
    raise ValueError(msg)
  return _read_only(thetas)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SimulationResult:
  r"""The sessions of a simulated run, with the run's tables and metrics (ADR-0016).

  :func:`simulate` returns one; one can also be built from sessions an application ran itself,
  whose true abilities may be unknown. A result is immutable: its sessions are a tuple of
  :class:`~catsim.Session` values, which never change, and its arrays are read-only. It compares by
  identity.

  **Tables.** :meth:`steps` has one row per session and step, :meth:`examinees` one row per session,
  :meth:`deciles` one row per decile of the true ability and :meth:`ability_levels` one row per
  distinct true ability. Each is a dict of equal-length NumPy columns, which
  ``pandas.DataFrame(result.steps())`` and polars take as they are, and which :class:`csv.DictWriter`
  writes row by row. Items appear by index and by name (ADR-0004).

  **Metrics.** :meth:`bias`, :meth:`mse`, :meth:`rmse`, :meth:`test_length`, :meth:`se`,
  :meth:`exposure_rates` and :meth:`overlap_rate` each return a :class:`~catsim.metrics.Metric`,
  the value and its Monte Carlo standard error, computed by :mod:`catsim.metrics` over the run's
  sessions, or with ``decile=k`` over the sessions of decile ``k`` only; :meth:`proposal_rates`
  covers the whole run. The Monte Carlo errors treat the sessions as independent: with exposure
  feedback (an ``r_max`` cap), each session depends on the sessions before it, so the errors are
  approximations, and replications of the run give the exact spread. A decile with no session
  gives NaN. Bias, MSE and RMSE, and every metric by decile, need the true abilities, and raise
  without them.

  **Deciles.** Decile :math:`k`, from 0 to 9, holds the sessions whose true ability :math:`\theta`
  has :math:`q_k \leq \theta < q_{k+1}`, where :math:`q_0, \ldots, q_{10}` are the quantiles of the
  run's true abilities at :math:`0, 0.1, \ldots, 1`, interpolated linearly (NumPy's default, R's
  type 7), and decile 9 also holds the abilities equal to :math:`q_{10}`, the highest. With
  :math:`N` distinct abilities, each decile holds :math:`N/10` sessions when 10 divides :math:`N`.
  Equal abilities always share a decile, so with ties some deciles may hold more sessions and others
  none: when every examinee has the same ability, all are in decile 9.

  The :doc:`/running-simulations` guide reads a result.

  Parameters
  ----------
  item_bank : ItemBank
      The item bank the sessions drew from.
  sessions : Sequence[Session]
      The run's sessions, at least one, in examinee order, each ended. Stored as a tuple.
  proposal_counts : array-like
      Each item's proposals over the run, as integers: its administrations plus the times the
      engine's exposure control declined it. A session proposes an item at most once, so a count is
      between the item's administrations and the number of sessions. Without an exposure control,
      the proposals are the administrations.
  duration : float
      The run's wall-clock time, in seconds.
  true_thetas : array-like or None, optional
      Each session's true ability, a finite number, in session order, or None when the true
      abilities are unknown, as for recorded examinees. A session does not hold its examinee's true
      ability, so the result does. Default is None.
  parts : int, optional
      The number of contiguous parts the run's examinees were split into, each with exposure counts
      of its own (ADR-0015). Default is 1, a sequential run.

  Attributes
  ----------
  exposure_counts : numpy.ndarray
      Each item's administrations over the run, counted from the sessions.

  Raises
  ------
  TypeError
      If ``item_bank`` is not an :class:`~catsim.ItemBank`, ``duration`` is not a number, a session
      is not a :class:`~catsim.Session`, the proposal counts are not integers, or ``parts`` is not
      an integer.
  ValueError
      If ``duration`` is negative or not finite, there is no session, a session has not ended or
      administered an index the bank does not have, ``true_thetas`` does not hold one finite number
      per session, the proposal counts are not one per item between its administrations and the
      number of sessions, or ``parts`` is not positive.

  Examples
  --------
  >>> import numpy
  >>> from catsim import CatEngine, ItemBank, simulate
  >>> from catsim.estimation import EAPEstimator
  >>> from catsim.initialization import FixedPointInitializer
  >>> from catsim.selection import MaxInfoSelector
  >>> from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule
  >>> engine = CatEngine(
  ...   ItemBank.generate(100, rng=0),
  ...   FixedPointInitializer(0.0),
  ...   MaxInfoSelector(),
  ...   EAPEstimator(),
  ...   StopRule(StandardErrorStopper(0.5), MaxLengthStopper(30)),
  ... )
  >>> result = simulate(engine, true_thetas=numpy.random.default_rng(1).normal(0, 1, 200), rng=0)
  >>> value, mc_se = result.rmse()
  >>> print(f"RMSE {value:.3f} ± {mc_se:.3f}, mean length {result.test_length().value:.1f}")
  RMSE 0.472 ± 0.025, mean length 13.9
  >>> list(result.steps())
  ['session', 'position', 'item', 'name', 'response', 'theta', 'se', 'evaluations', 'true_theta']
  >>> by_decile = result.deciles()
  >>> by_decile["n_sessions"].tolist()
  [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
  >>> by_decile["test_length"].round(1).tolist()
  [14.8, 13.4, 13.3, 14.0, 13.2, 13.2, 14.2, 14.1, 14.4, 14.8]

  The tests differ in length, so there is no overlap rate:

  >>> result.overlap_rate() is None
  True
  """

  item_bank: ItemBank
  sessions: tuple[Session, ...]
  proposal_counts: IntArray
  duration: float
  true_thetas: FloatArray | None = None
  parts: int = 1
  exposure_counts: IntArray = field(init=False)

  def __post_init__(self) -> None:
    """Store the sessions as a tuple and the arrays read-only, after checking that they belong together."""
    if not isinstance(self.item_bank, ItemBank):
      msg = f"item_bank is a {type(self.item_bank).__name__}; it must be an ItemBank"
      raise TypeError(msg)
    # A bool is an int, and True passed as a duration is a flag passed by mistake.
    if isinstance(self.duration, (bool, numpy.bool_)) or not isinstance(self.duration, (int, float, numpy.number)):
      msg = f"duration is {self.duration!r}; it must be a number of seconds"
      raise TypeError(msg)
    if not (math.isfinite(self.duration) and self.duration >= 0):
      msg = f"duration is {self.duration!r}; a run's wall-clock time is a finite, non-negative number of seconds"
      raise ValueError(msg)
    sessions = _finished_sessions(self.sessions)
    n_items = self.item_bank.n_items
    _check_items(sessions, n_items)
    exposure_counts = _read_only(_counts(sessions, n_items))
    object.__setattr__(self, "sessions", sessions)
    object.__setattr__(self, "exposure_counts", exposure_counts)
    proposal_counts = _proposal_counts(self.proposal_counts, exposure_counts, len(sessions), self.item_bank.names)
    object.__setattr__(self, "proposal_counts", proposal_counts)
    if self.true_thetas is not None:
      object.__setattr__(self, "true_thetas", _result_true_thetas(self.true_thetas, len(sessions)))
    if isinstance(self.parts, (bool, numpy.bool_)) or not isinstance(self.parts, (int, numpy.integer)):
      msg = f"parts is {self.parts!r}; it must be an integer"
      raise TypeError(msg)
    if self.parts < 1:
      msg = f"parts is {self.parts}; a run has at least one part"
      raise ValueError(msg)

  @override
  def __reduce__(self) -> tuple[functools.partial[SimulationResult], tuple[()]]:
    """Return how to rebuild the result through its constructor, which makes the copy's arrays read-only.

    NumPy pickles an array without its read-only flag, so a pickled or deep-copied result is built
    anew from its fields rather than restored from them.

    Returns
    -------
    tuple
        A call that builds an equal result, and its empty positional arguments.
    """
    return functools.partial(
      SimulationResult,
      item_bank=self.item_bank,
      sessions=self.sessions,
      proposal_counts=self.proposal_counts,
      duration=self.duration,
      true_thetas=self.true_thetas,
      parts=self.parts,
    ), ()

  def steps(self) -> dict[str, numpy.ndarray]:
    """Return the table of the run's steps: one row per session and administered item.

    The rows are in session order, and in the order the items were given within a session.

    Returns
    -------
    dict[str, numpy.ndarray]
        The columns, each with one entry per row:

        - ``session`` (int): the session's position in the run, its examinee's position;
        - ``position`` (int): the item's position in its session's test, 0 for the first, so the
          row's item is ``sessions[session].administered_items[position]``;
        - ``item`` (int): the item's index in the bank;
        - ``name`` (str): the item's name, ``item_bank.names[item]``;
        - ``response`` (bool): the examinee's response;
        - ``theta`` (float): the ability estimate after the response;
        - ``se`` (float): its standard error;
        - ``evaluations`` (float): how many times the estimator evaluated its objective for that
          estimate, NaN where the estimator gave no count (:class:`~catsim.Estimate`);
        - ``true_theta`` (float): the examinee's true ability, a column only when the result has
          true abilities.

        Each session's initial estimate, which precedes its first item, is the first of the
        session's ``estimates``.
    """
    lengths = self._test_lengths()
    total = int(lengths.sum())
    estimates = [estimate for session in self.sessions for estimate in session.estimates[1:]]
    items = numpy.fromiter(
      itertools.chain.from_iterable(session.administered_items for session in self.sessions), dtype=int, count=total
    )
    table = {
      "session": numpy.repeat(numpy.arange(len(self.sessions)), lengths),
      "position": numpy.arange(total) - numpy.repeat(numpy.cumsum(lengths) - lengths, lengths),
      "item": items,
      "name": numpy.array(self.item_bank.names)[items],
      "response": numpy.fromiter(
        itertools.chain.from_iterable(session.responses for session in self.sessions), dtype=bool, count=total
      ),
      "theta": numpy.array([estimate.theta for estimate in estimates], dtype=float),
      "se": numpy.array([estimate.se for estimate in estimates], dtype=float),
      "evaluations": numpy.array(
        [math.nan if estimate.evaluations is None else estimate.evaluations for estimate in estimates], dtype=float
      ),
    }
    if self.true_thetas is not None:
      table["true_theta"] = numpy.repeat(self.true_thetas, lengths)
    return table

  def examinees(self) -> dict[str, numpy.ndarray]:
    """Return the table of the run's examinees: one row per session, in session order.

    Returns
    -------
    dict[str, numpy.ndarray]
        The columns, each with one entry per session:

        - ``session`` (int): the session's position in the run, its examinee's position;
        - ``test_length`` (int): how many items the session administered;
        - ``theta`` (float): the session's final ability estimate;
        - ``se`` (float): its standard error, infinite for a session that ended before its first
          response;
        - ``stop_reason`` (str): why the session ended, the value of its
          :class:`~catsim.StopReason`, ``"stopper"`` or ``"no_items"``;
        - ``true_theta`` (float) and ``decile`` (int): the examinee's true ability and its decile,
          from 0 to 9, columns only when the result has true abilities.
    """
    table = {
      "session": numpy.arange(len(self.sessions)),
      "test_length": self._test_lengths(),
      "theta": self._final_thetas(),
      "se": self._final_ses(),
      "stop_reason": numpy.array([str(session.stop_reason) for session in self.sessions]),
    }
    if self.true_thetas is not None:
      table["true_theta"] = self.true_thetas.copy()
      table["decile"] = _deciles(self.true_thetas)
    return table

  def deciles(self) -> dict[str, numpy.ndarray]:
    """Return the table of the run's metrics by decile of the true ability: one row per decile, from the lowest.

    Returns
    -------
    dict[str, numpy.ndarray]
        Ten rows, with the columns:

        - ``decile`` (int): 0 to 9;
        - ``n_sessions`` (int): how many sessions the decile holds;
        - ``true_theta`` (float): the mean true ability of its sessions;
        - ``bias``, ``mse``, ``rmse``, ``test_length``, ``se`` and ``overlap_rate`` (float): the
          metrics of the methods of those names with ``decile`` set, each followed by its Monte
          Carlo error, ``bias_mc_se`` to ``overlap_rate_mc_se``. The overlap rate is NaN for a
          decile whose tests differ in length.

        A decile with no session has NaN for every float.

    Raises
    ------
    ValueError
        If the result has no true abilities.
    """
    groups = _deciles(self._known_true_thetas("deciles"))
    members = [numpy.flatnonzero(groups == decile) for decile in range(_N_DECILES)]
    return {"decile": numpy.arange(_N_DECILES), **self._group_table(members)}

  def ability_levels(self) -> dict[str, numpy.ndarray]:
    """Return the table of the run's metrics at each distinct true ability: one row per ability, from the lowest.

    It suits designs that give many examinees each of a few true abilities, such as 100 at each of
    13 abilities from -3 to 3, which :meth:`deciles` would group unevenly: 13 abilities do not fit
    in 10 deciles. Equal abilities are those that compare equal.

    Returns
    -------
    dict[str, numpy.ndarray]
        One row per distinct true ability, with the columns of :meth:`deciles` but ``decile``:
        ``n_sessions``, ``true_theta`` (the ability itself), and the metrics and their Monte Carlo
        errors. An ability held by a single session has NaN Monte Carlo errors.

    Raises
    ------
    ValueError
        If the result has no true abilities.
    """
    true_thetas = self._known_true_thetas("ability_levels")
    levels = numpy.unique(true_thetas)
    return self._group_table([numpy.flatnonzero(true_thetas == level) for level in levels])

  def bias(self, *, decile: int | None = None) -> Metric[float]:
    """Return the mean of the final estimates minus the true abilities, with its Monte Carlo error.

    See :func:`catsim.metrics.bias`.

    Parameters
    ----------
    decile : int or None, optional
        The decile of the true ability whose sessions to cover, 0 to 9, or None, the default, for
        every session.

    Returns
    -------
    Metric
        The bias and its Monte Carlo standard error.

    Raises
    ------
    ValueError
        If the result has no true abilities, or ``decile`` is not one of 0 to 9.
    """
    return self._accuracy(metrics.bias, "bias", decile)

  def mse(self, *, decile: int | None = None) -> Metric[float]:
    """Return the mean squared difference of the final estimates from the true abilities, with its Monte Carlo error.

    See :func:`catsim.metrics.mse`.

    Parameters
    ----------
    decile : int or None, optional
        The decile of the true ability whose sessions to cover, 0 to 9, or None, the default, for
        every session.

    Returns
    -------
    Metric
        The mean squared error and its Monte Carlo standard error.

    Raises
    ------
    ValueError
        If the result has no true abilities, or ``decile`` is not one of 0 to 9.
    """
    return self._accuracy(metrics.mse, "mse", decile)

  def rmse(self, *, decile: int | None = None) -> Metric[float]:
    """Return the root of :meth:`mse`, with its Monte Carlo error.

    See :func:`catsim.metrics.rmse`.

    Parameters
    ----------
    decile : int or None, optional
        The decile of the true ability whose sessions to cover, 0 to 9, or None, the default, for
        every session.

    Returns
    -------
    Metric
        The root mean squared error and its Monte Carlo standard error.

    Raises
    ------
    ValueError
        If the result has no true abilities, or ``decile`` is not one of 0 to 9.
    """
    return self._accuracy(metrics.rmse, "rmse", decile)

  def test_length(self, *, decile: int | None = None) -> Metric[float]:
    """Return the mean number of items the sessions administered, with its Monte Carlo error.

    See :func:`catsim.metrics.mean`.

    Parameters
    ----------
    decile : int or None, optional
        The decile of the true ability whose sessions to cover, 0 to 9, or None, the default, for
        every session.

    Returns
    -------
    Metric
        The mean test length and its Monte Carlo standard error.

    Raises
    ------
    ValueError
        If ``decile`` is given and the result has no true abilities, or ``decile`` is not one of 0
        to 9.
    """
    lengths = self._test_lengths()
    return self._over_sessions(decile, lambda rows: metrics.mean(lengths[rows]))

  def se(self, *, decile: int | None = None) -> Metric[float]:
    """Return the mean standard error of the sessions' final estimates, with its Monte Carlo error.

    See :func:`catsim.metrics.mean`. It is infinite, with a NaN error, when a session ended before
    its first response.

    Parameters
    ----------
    decile : int or None, optional
        The decile of the true ability whose sessions to cover, 0 to 9, or None, the default, for
        every session.

    Returns
    -------
    Metric
        The mean standard error and its Monte Carlo standard error.

    Raises
    ------
    ValueError
        If ``decile`` is given and the result has no true abilities, or ``decile`` is not one of 0
        to 9.
    """
    ses = self._final_ses()
    return self._over_sessions(decile, lambda rows: metrics.mean(ses[rows]))

  def exposure_rates(self, *, decile: int | None = None) -> Metric[FloatArray]:
    """Return each item's exposure rate, the share of the sessions that were given it, with its Monte Carlo error.

    See :func:`catsim.metrics.exposure_rates`. These are the rates at the end of the run, over all
    its sessions, not the rates any session saw (ADR-0012).

    Parameters
    ----------
    decile : int or None, optional
        The decile of the true ability whose sessions to cover, 0 to 9, or None, the default, for
        every session.

    Returns
    -------
    Metric
        One rate and one Monte Carlo standard error per item, as arrays.

    Raises
    ------
    ValueError
        If ``decile`` is given and the result has no true abilities, or ``decile`` is not one of 0
        to 9.
    """
    rows = self._rows(decile)
    if not rows.size:
      empty = numpy.full(self.item_bank.n_items, math.nan)
      return Metric(empty, empty.copy())
    counts = _counts([self.sessions[row] for row in rows], self.item_bank.n_items)
    return metrics.exposure_rates(counts, int(rows.size))

  def proposal_rates(self) -> Metric[FloatArray]:
    """Return the share of the sessions in which the selector proposed each item, with its Monte Carlo error.

    A session proposes an item at most once, as the exposure control declines it or the session
    administers it, so the rate is at most 1. Without an exposure control it equals
    :meth:`exposure_rates`. See :func:`catsim.metrics.exposure_rates`.

    Returns
    -------
    Metric
        One rate and one Monte Carlo standard error per item, as arrays.
    """
    return metrics.exposure_rates(self.proposal_counts, len(self.sessions))

  def overlap_rate(self, *, decile: int | None = None) -> Metric[float] | None:
    """Return the test overlap rate, with its Monte Carlo error, or None unless every session gave as many items.

    See :func:`catsim.metrics.overlap_rate`, whose formula assumes one fixed test length, so there is
    no overlap rate for tests of different lengths, or of no items. By decile, it is the overlap of
    the tests of examinees of similar ability.

    Parameters
    ----------
    decile : int or None, optional
        The decile of the true ability whose sessions to cover, 0 to 9, or None, the default, for
        every session.

    Returns
    -------
    Metric or None
        The overlap rate and its Monte Carlo standard error, or None.

    Raises
    ------
    ValueError
        If ``decile`` is given and the result has no true abilities, or ``decile`` is not one of 0
        to 9.
    """
    rows = self._rows(decile)
    return self._overlap(rows) if rows.size else Metric(math.nan, math.nan)

  def _overlap(self, rows: IntArray) -> Metric[float] | None:
    """Return the overlap rate of the sessions at ``rows``, or None unless they all gave as many items, at least one."""
    tests = [self.sessions[row].administered_items for row in rows]
    if not tests[0] or any(len(test) != len(tests[0]) for test in tests):
      return None
    return metrics.overlap_rate(numpy.array(tests), self.item_bank.n_items)

  def _group_table(self, groups: Sequence[IntArray]) -> dict[str, numpy.ndarray]:
    """Return the size, mean true ability and metrics of each group of sessions, given by positions.

    A group with no session has NaN for every float, and the overlap rate is NaN for a group whose
    tests differ in length.

    Parameters
    ----------
    groups : Sequence[numpy.ndarray]
        The positions of each group's sessions.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``n_sessions``, ``true_theta``, then each metric followed by its Monte Carlo error.
    """
    true_thetas = cast("FloatArray", self.true_thetas)
    thetas, lengths, ses = self._final_thetas(), self._test_lengths(), self._final_ses()
    columns: dict[str, Callable[[IntArray], Metric[float] | None]] = {
      "bias": lambda rows: metrics.bias(true_thetas[rows], thetas[rows]),
      "mse": lambda rows: metrics.mse(true_thetas[rows], thetas[rows]),
      "rmse": lambda rows: metrics.rmse(true_thetas[rows], thetas[rows]),
      "test_length": lambda rows: metrics.mean(lengths[rows]),
      "se": lambda rows: metrics.mean(ses[rows]),
      "overlap_rate": self._overlap,
    }
    table = {
      "n_sessions": numpy.array([rows.size for rows in groups], dtype=int),
      "true_theta": numpy.array([true_thetas[rows].mean() if rows.size else math.nan for rows in groups]),
    }
    for name, metric in columns.items():
      results = [metric(rows) if rows.size else None for rows in groups]
      values = [Metric(math.nan, math.nan) if result is None else result for result in results]
      table[name] = numpy.array([value.value for value in values], dtype=float)
      table[f"{name}_mc_se"] = numpy.array([value.mc_se for value in values], dtype=float)
    return table

  def _test_lengths(self) -> IntArray:
    """Return how many items each session administered."""
    return numpy.array([len(session.administered_items) for session in self.sessions], dtype=int)

  def _final_thetas(self) -> FloatArray:
    """Return each session's final ability estimate."""
    return numpy.array([session.current_theta for session in self.sessions], dtype=float)

  def _final_ses(self) -> FloatArray:
    """Return the standard error of each session's final ability estimate."""
    return numpy.array([session.current_se for session in self.sessions], dtype=float)

  def _known_true_thetas(self, needed_by: str) -> FloatArray:
    """Return the true abilities, raising if the result has none.

    Parameters
    ----------
    needed_by : str
        What needs them, for the error message.

    Returns
    -------
    numpy.ndarray
        Each session's true ability.

    Raises
    ------
    ValueError
        If the result's true abilities are unknown.
    """
    if self.true_thetas is None:
      msg = (
        f"{needed_by} needs the examinees' true abilities, and this result has none; pass true_thetas to simulate "
        "with the responses when they are known"
      )
      raise ValueError(msg)
    return self.true_thetas

  def _rows(self, decile: int | None) -> IntArray:
    """Return the positions of the sessions a metric covers: every session, or those of one decile.

    Parameters
    ----------
    decile : int or None
        The decile, or None for every session.

    Returns
    -------
    numpy.ndarray
        The sessions' positions, in order.

    Raises
    ------
    TypeError
        If ``decile`` is neither None nor an integer.
    ValueError
        If ``decile`` is not one of 0 to 9, or the result has no true abilities.
    """
    if decile is None:
      return numpy.arange(len(self.sessions))
    if isinstance(decile, (bool, numpy.bool_)) or not isinstance(decile, (int, numpy.integer)):
      msg = f"decile is {decile!r}; it must be None or an integer from 0 to 9"
      raise TypeError(msg)
    if not 0 <= decile < _N_DECILES:
      msg = f"decile is {decile}; the deciles are 0, of the lowest true abilities, to 9"
      raise ValueError(msg)
    return numpy.flatnonzero(_deciles(self._known_true_thetas("a metric by decile")) == decile)

  def _over_sessions(self, decile: int | None, metric: Callable[[IntArray], Metric[float]]) -> Metric[float]:
    """Return a metric over the sessions it covers, NaN for a decile with none.

    Parameters
    ----------
    decile : int or None
        The decile, or None for every session.
    metric : Callable[[numpy.ndarray], Metric]
        Computes the metric from the positions of the sessions it covers.

    Returns
    -------
    Metric
        The metric and its Monte Carlo standard error.
    """
    rows = self._rows(decile)
    return metric(rows) if rows.size else Metric(math.nan, math.nan)

  def _accuracy(
    self, metric: Callable[[FloatArray, FloatArray], Metric[float]], name: str, decile: int | None
  ) -> Metric[float]:
    """Return a metric of the final estimates' accuracy over the sessions it covers.

    Parameters
    ----------
    metric : Callable[[numpy.ndarray, numpy.ndarray], Metric]
        A function of :mod:`catsim.metrics` that takes the true abilities and the estimates.
    name : str
        The metric's name, for the error message.
    decile : int or None
        The decile, or None for every session.

    Returns
    -------
    Metric
        The metric and its Monte Carlo standard error.
    """
    true_thetas = self._known_true_thetas(name)
    thetas = self._final_thetas()
    return self._over_sessions(decile, lambda rows: metric(true_thetas[rows], thetas[rows]))
