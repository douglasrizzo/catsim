"""Simulation runner built on top of the explicit CAT engine."""

from __future__ import annotations

import time
from typing import Any

import numpy
import numpy.typing as npt
from tqdm import tqdm

from .engine import CatEngine, RunContext, SimulatedResponseProvider
from .estimation import BaseEstimator
from .initialization import BaseInitializer
from .item_bank import ItemBank
from .selection import BaseSelector
from .state import ExposureTracker, SimulationResult
from .stopping import BaseStopper


class SimulationRunner:
  """Run CAT simulations for one or more examinees."""

  def __init__(
    self,
    item_bank: ItemBank | npt.NDArray[numpy.floating[Any]],
    initializer: BaseInitializer,
    selector: BaseSelector,
    estimator: BaseEstimator,
    stopper: BaseStopper,
    seed: int = 0,
  ) -> None:
    if isinstance(item_bank, numpy.ndarray):
      item_bank = ItemBank(item_bank)
    elif not isinstance(item_bank, ItemBank):
      msg = "item_bank must be an ItemBank or numpy.ndarray"
      raise TypeError(msg)

    self._item_bank = item_bank
    self._engine = CatEngine(initializer, selector, estimator, stopper)
    self._rng = numpy.random.default_rng(seed=seed)

  @property
  def item_bank(self) -> ItemBank:
    """Return the configured item bank."""
    return self._item_bank

  @property
  def rng(self) -> numpy.random.Generator:
    """Return the simulation RNG."""
    return self._rng

  def _to_distribution(self, examinees: int | npt.ArrayLike) -> npt.NDArray[numpy.floating[Any]]:
    if isinstance(examinees, int):
      if examinees <= 0:
        msg = f"Number of examinees must be positive, got {examinees}"
        raise ValueError(msg)
      mean = numpy.mean(self._item_bank.difficulty)
      stddev = numpy.std(self._item_bank.difficulty)
      return self._rng.normal(mean, stddev, examinees)

    x_array = numpy.asarray(examinees)
    if x_array.ndim != 1:
      msg = "Examinees array must be one-dimensional"
      raise TypeError(msg)
    if x_array.size == 0:
      msg = "Array of examinees cannot be empty"
      raise ValueError(msg)
    return x_array

  def run(self, examinees: int | npt.ArrayLike, verbose: bool = False) -> SimulationResult:
    """Run a CAT simulation for the given examinees."""
    thetas = self._to_distribution(examinees)
    tracker = ExposureTracker(self._item_bank.n_items)
    context = RunContext(rng=self._rng, exposure_tracker=tracker, total_sessions=len(thetas))
    response_provider = SimulatedResponseProvider()
    sessions = []

    pbar = tqdm(total=len(thetas)) if verbose else None
    start_time = time.time()

    for session_id, true_theta in enumerate(thetas):
      session = self._engine.start_session(
        session_id=session_id,
        item_bank=self._item_bank,
        context=context,
        true_theta=float(true_theta),
      )
      session = self._engine.run_session(session, self._item_bank, response_provider, context)
      sessions.append(session)
      if pbar is not None:
        pbar.update()

    if pbar is not None:
      pbar.close()

    duration = time.time() - start_time
    return SimulationResult(
      item_bank=self._item_bank,
      sessions=sessions,
      exposure_counts=tracker.counts.copy(),
      duration=duration,
    )
