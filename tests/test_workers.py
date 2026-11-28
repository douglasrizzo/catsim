"""Tests for a run's parts on worker processes, and the check that the workers can import the engine (ADR-0015)."""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
import threading
from concurrent.futures.process import BrokenProcessPool
from types import ModuleType
from typing import TYPE_CHECKING, override

import numpy as np
import pytest
from irtlib import ItemBank
from irtlib.typing import FloatArray

from catsim import CatEngine, StepContext, simulate
from catsim.estimation import EAPEstimator, MAPEstimator
from catsim.initialization import FixedPointInitializer, UniformInitializer
from catsim.selection import BaseSelector, MaxInfoSelector, MEISelector, ProgressiveSelector
from catsim.stopping import MaxLengthStopper

if TYPE_CHECKING:
  from collections.abc import Iterator


class MainModuleSelector(MaxInfoSelector):
  """Maximum information, from a class that says, as one defined in a notebook does, that the main module defines it."""


MainModuleSelector.__module__ = "__main__"


def main_module_log_prior(theta: FloatArray) -> FloatArray:
  """Return a standard normal log-density, up to a constant, from a function that says the main module defines it."""
  return -0.5 * np.asarray(theta) ** 2


main_module_log_prior.__module__ = "__main__"


class MainModuleEngine(CatEngine):
  """An engine class that says, as one defined in a notebook does, that the main module defines it."""


MainModuleEngine.__module__ = "__main__"

# Strategies that fail in a worker, for a module the workers import from sys.path: they cannot import this module,
# which pytest imports from its path.
FAILING_STRATEGIES = '''
"""Selectors that fail in a worker process."""

import os

from catsim.selection import MaxInfoSelector


class RaisingSelector(MaxInfoSelector):
    """Maximum information for the first item, then an error."""

    def select(self, context):
        if context.administered_items:
            raise RuntimeError("the selector failed at the second item")
        return super().select(context)


class ExitingSelector(MaxInfoSelector):
    """A selector whose process exits at once, as one killed by the system would."""

    def select(self, context):
        os._exit(1)
'''


class LockedSelector(MaxInfoSelector):
  """Maximum information under a lock, which pickle cannot send."""

  def __init__(self) -> None:
    super().__init__()
    self._lock = threading.Lock()

  @override
  def select(self, context: StepContext) -> int:
    with self._lock:
      return super().select(context)


def engine_with(
  *, selector: BaseSelector | None = None, estimator: EAPEstimator | MAPEstimator | None = None
) -> CatEngine:
  """Return an engine of four-item tests on a 30-item bank, with maximum information and EAP unless told otherwise."""
  return CatEngine(
    ItemBank.generate(30, rng=4),
    FixedPointInitializer(0.0),
    selector or MaxInfoSelector(),
    estimator or EAPEstimator(),
    MaxLengthStopper(4),
  )


@pytest.fixture(scope="module")
def failing_strategies(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ModuleType]:
  """Provide the module of FAILING_STRATEGIES, written to a directory on sys.path, where the workers import it."""
  directory = tmp_path_factory.mktemp("strategies")
  (directory / "failing_strategies.py").write_text(FAILING_STRATEGIES, encoding="utf-8")
  with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(directory))
    yield importlib.import_module("failing_strategies")
  sys.modules.pop("failing_strategies", None)


@pytest.mark.parametrize(
  ("workers", "recorded"),
  [(2, False), (3, False), (2, True)],
  ids=["two-workers", "three-workers", "two-workers-recorded-examinees"],
)
def test_the_workers_do_not_change_the_output(workers: int, recorded: bool) -> None:
  """Four parts on two or three workers give the run of the same four parts in the calling process.

  The initializer and the progressive selector draw from the sessions' generators, and the selector's exposure cap
  reads each part's counts. Some recorded answers are missing. Fails if the workers split the examinees by their own
  number rather than by ``parts``, share exposure counts, put the parts back in the order they finish, lose the
  global position of a part's examinees, or slice the recorded rows unlike the true abilities.
  """
  bank = ItemBank.generate(40, rng=1)
  engine = CatEngine(
    bank,
    UniformInitializer(-1.0, 1.0),
    ProgressiveSelector(test_size=6, r_max=0.4),
    EAPEstimator(),
    MaxLengthStopper(6),
  )
  thetas = np.random.default_rng(2).normal(0, 1, 13)
  if recorded:
    rows = (np.random.default_rng(3).random((13, bank.n_items)) < 0.5).astype(float)
    rows[np.random.default_rng(4).random(rows.shape) < 0.2] = np.nan
    examinees = {"responses": rows}
  else:
    examinees = {"true_thetas": thetas}

  in_process = simulate(engine, rng=6, parts=4, **examinees)  # ty: ignore[invalid-argument-type]
  on_workers = simulate(engine, rng=6, parts=4, workers=workers, **examinees)  # ty: ignore[invalid-argument-type]

  assert on_workers.sessions == in_process.sessions
  assert on_workers.exposure_counts.tolist() == in_process.exposure_counts.tolist()
  assert on_workers.proposal_counts is not None
  assert in_process.proposal_counts is not None
  assert on_workers.proposal_counts.tolist() == in_process.proposal_counts.tolist()
  assert on_workers.parts == 4


@pytest.mark.parametrize(
  ("engine", "message"),
  [
    pytest.param(
      engine_with(selector=MainModuleSelector()),
      r"the engine's selector \(MainModuleSelector\) refers to the class MainModuleSelector, defined in the main "
      r"module \(a notebook, an interactive session or the script being run\), which they cannot import\. Move "
      r"MainModuleSelector into a module",
      id="notebook-selector",
    ),
    pytest.param(
      engine_with(estimator=EAPEstimator(log_prior=main_module_log_prior)),
      r"the engine's estimator \(EAPEstimator\) refers to the function main_module_log_prior, defined in the main "
      r"module",
      id="notebook-prior",
    ),
    pytest.param(
      engine_with(estimator=EAPEstimator(log_prior=lambda theta: -0.5 * theta**2)),
      r"the engine's estimator \(EAPEstimator\) refers to a lambda of module [\w.]+, which they cannot import\. "
      r"Define it with def at the top level of a module and import it, or run with workers=1",
      id="lambda-prior",
    ),
    pytest.param(
      engine_with(selector=MEISelector(estimator=MAPEstimator(log_prior=lambda theta: -0.5 * theta**2))),
      r"the engine's selector \(MEISelector\) refers to a lambda",
      id="lambda-prior-of-the-look-ahead-estimator",
    ),
    pytest.param(
      MainModuleEngine(
        ItemBank.generate(30, rng=4), FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), MaxLengthStopper(4)
      ),
      r"the engine \(MainModuleEngine\) refers to the class MainModuleEngine, defined in the main module",
      id="notebook-engine",
    ),
    pytest.param(
      engine_with(selector=LockedSelector()),
      r"the engine's selector \(LockedSelector\) cannot be pickled \(cannot pickle '_thread\.lock' object\)\. Run "
      r"with workers=1",
      id="unpicklable-selector",
    ),
  ],
)
def test_an_engine_the_workers_cannot_import_raises_before_any_part_runs(engine: CatEngine, message: str) -> None:
  """A strategy the spawned workers could not import fails in the calling process, naming it and what to do.

  A class or function of the main module is sent by a name that the workers' main module lacks, and a lambda by none
  at all. Fails if the engine reaches the workers unchecked, where the import fails with an AttributeError that names
  no strategy (or pickle's own error, for the classes pickle cannot find in this process), if a strategy held by
  another one, or the engine's own class, is not checked, or if a session runs before the error.
  """
  calls: list[int] = []

  with pytest.raises(ValueError, match=message):
    simulate(
      engine, true_thetas=[0.0, 1.0, -1.0], rng=0, parts=3, workers=2, progress=lambda done, _: calls.append(done)
    )

  assert calls == []


def test_a_local_function_raises_before_any_part_runs() -> None:
  """A prior defined inside a function has no name a worker could import; fails if it reaches the workers unchecked."""

  def local_log_prior(theta: FloatArray) -> FloatArray:
    return -0.5 * np.asarray(theta) ** 2

  engine = engine_with(estimator=MAPEstimator(log_prior=local_log_prior))

  with pytest.raises(
    ValueError,
    match=r"the engine's estimator \(MAPEstimator\) refers to the function "
    r"test_a_local_function_raises_before_any_part_runs\.<locals>\.local_log_prior, defined inside a function",
  ):
    simulate(engine, true_thetas=[0.0, 1.0], rng=0, parts=2, workers=2)


@pytest.mark.parametrize(("parts", "workers"), [(2, 1), (1, 4)], ids=["one-worker", "one-part"])
def test_strategies_the_workers_cannot_import_run_in_the_calling_process(parts: int, workers: int) -> None:
  """With one worker, or one part for any number of workers, no process starts, so a notebook's strategies run.

  Without an exposure cap, the run gives the sessions of a run with default arguments. Fails if the engine is checked,
  or sent to a process, when no part runs on a worker.
  """
  engine = engine_with(selector=MainModuleSelector(), estimator=EAPEstimator(log_prior=lambda theta: -0.5 * theta**2))
  thetas = [-1.0, 0.0, 1.0]

  result = simulate(engine, true_thetas=thetas, rng=3, parts=parts, workers=workers)

  assert result.examinees()["test_length"].tolist() == [4, 4, 4]
  assert result.sessions == simulate(engine, true_thetas=thetas, rng=3).sessions


def test_a_class_defined_in_an_interactive_session_raises_before_any_part_runs() -> None:
  """A selector class defined in ``python -c``, whose main module has no file, as a notebook's has none, is refused.

  The spawned workers cannot import such a class, so without the check its run fails in the workers with an
  AttributeError that names no strategy, after the calling process has started them.
  """
  code = textwrap.dedent(
    """
    from catsim import CatEngine, simulate
    from irtlib import ItemBank
    from catsim.estimation import EAPEstimator
    from catsim.initialization import FixedPointInitializer
    from catsim.selection import MaxInfoSelector
    from catsim.stopping import MaxLengthStopper

    class NotebookSelector(MaxInfoSelector):
        pass

    bank = ItemBank.generate(20, rng=0)
    engine = CatEngine(bank, FixedPointInitializer(0.0), NotebookSelector(), EAPEstimator(), MaxLengthStopper(3))
    try:
        simulate(engine, true_thetas=[0.0, 1.0], rng=0, parts=2, workers=2, progress=lambda *_: print("FINISHED"))
    except ValueError as error:
        print(error)
    """
  )

  completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)

  assert completed.returncode == 0, completed.stderr
  assert "FINISHED" not in completed.stdout
  assert (
    "the engine's selector (NotebookSelector) refers to the class NotebookSelector, defined in the main module"
    in completed.stdout
  )


@pytest.mark.parametrize("report", [False, True], ids=["without-progress", "with-progress"])
def test_a_part_that_fails_on_a_worker_raises_its_error_in_the_calling_process(
  failing_strategies: ModuleType, report: bool
) -> None:
  """A strategy's exception in a worker reaches the caller as itself, whether or not the workers report progress.

  Fails if the calling process waits forever for the failed part's sessions, or raises another error in its place.
  """
  engine = engine_with(selector=failing_strategies.RaisingSelector())

  with pytest.raises(RuntimeError, match="the selector failed at the second item"):
    simulate(
      engine,
      true_thetas=[0.0, 1.0, 2.0, -1.0],
      rng=0,
      parts=2,
      workers=2,
      progress=(lambda *_: None) if report else None,
    )


def test_a_worker_that_dies_breaks_the_run_rather_than_hanging_it(failing_strategies: ModuleType) -> None:
  """A worker that exits in the middle of its part never says its task has ended.

  Fails if the calling process, which reads the workers' progress, waits for that message forever rather than raising
  the pool's BrokenProcessPool.
  """
  engine = engine_with(selector=failing_strategies.ExitingSelector())

  with pytest.raises(BrokenProcessPool):
    simulate(engine, true_thetas=[0.0, 1.0], rng=0, parts=2, workers=2, progress=lambda *_: None)


def test_a_progress_callback_that_raises_on_workers_stops_the_run() -> None:
  """The error of progress, called in the calling process as the workers report, leaves simulate as it is.

  Fails if the relay of the workers' messages swallows it, or replaces it with an error of its own.
  """

  def progress(finished: int, total: int) -> None:
    msg = f"progress failed at {finished} of {total}"
    raise LookupError(msg)

  with pytest.raises(LookupError, match="progress failed at 1 of 4"):
    simulate(engine_with(), true_thetas=[0.0, 1.0, 2.0, -1.0], rng=0, parts=2, workers=2, progress=progress)
