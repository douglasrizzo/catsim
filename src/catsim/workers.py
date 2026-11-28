"""The worker processes that run the parts of a parallel run, and the check that they can receive an engine.

:func:`catsim.simulate` runs a run's parts here when it has more than one part and more than one
worker (ADR-0015). Each worker receives the engine pickled, and unpickling imports every class and
function the engine refers to by its module and name, so the engine is checked first: a strategy
defined in the main module (a notebook, an interactive session or the script being run), a lambda
or a function defined inside another function raises before any part runs.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import io
import multiprocessing
import pickle  # noqa: S403 - the workers unpickle only what the calling process pickled
import queue
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence
  from concurrent.futures import Future
  from multiprocessing.queues import Queue

  from .engine import CatEngine


# How long the calling process waits for a message before it checks whether a task has failed.
_POLL_SECONDS = 0.1

# The most processes a process pool may have on Windows, where concurrent.futures raises for more.
_MAX_WINDOWS_WORKERS = 61

# The worker's end of the run's message queue, which its initializer sets; None when nobody listens.
_worker: dict[str, Queue[_Message] | None] = {"messages": None}


class _Message(enum.Enum):
  """What a worker tells the calling process."""

  SESSION = "session"  # a session of the task has ended
  TASK = "task"  # the task has returned or raised, and sends nothing more


class _UnimportableError(Exception):
  """Raised by :class:`_ImportCheck` at the first class or function a worker could not import by its name."""

  def __init__(self, description: str, advice: str) -> None:
    super().__init__(description)
    self.description = description
    self.advice = advice


class _ImportCheck(pickle.Pickler):
  """A pickler that raises :class:`_UnimportableError` at a class or function a worker could not import.

  Pickle sends a class or a function by its module and qualified name, which the receiving process
  imports. The worker processes start afresh (they are spawned), so they import the main module
  under another name and without what a notebook or an interactive session defined in it, and no
  name leads to a lambda or to a function or class defined inside a function.
  """

  @override
  def reducer_override(self, obj: object) -> str | tuple[Any, ...]:
    """Raise at a class or function a worker could not import, and leave every object to the usual pickling.

    Parameters
    ----------
    obj : object
        The object being pickled.

    Returns
    -------
    object
        ``NotImplemented``, so that pickle reduces ``obj`` as usual.

    Raises
    ------
    _UnimportableError
        If ``obj`` is a class or function defined in the main module, a lambda, or a class or
        function defined inside a function.
    """
    if isinstance(obj, (type, types.FunctionType)) and (problem := _unimportable(obj)) is not None:
      raise _UnimportableError(*problem)
    return NotImplemented


def _unimportable(obj: type | types.FunctionType) -> tuple[str, str] | None:
  """Return why a worker could not import a class or function, and what to do about it, or None if it could.

  Parameters
  ----------
  obj : type or function
      The class or function.

  Returns
  -------
  tuple[str, str] or None
      What ``obj`` is, for the error message, and the advice, or None.
  """
  kind = "class" if isinstance(obj, type) else "function"
  name = obj.__qualname__
  if "<lambda>" in name:
    return f"a lambda of module {obj.__module__}", "Define it with def at the top level of a module and import it"
  if "<locals>" in name:
    return f"the {kind} {name}, defined inside a function", "Move it to the top level of a module and import it"
  if obj.__module__ == "__main__":
    return (
      f"the {kind} {name}, defined in the main module (a notebook, an interactive session or the script being run)",
      f"Move {name} into a module, such as a .py file beside the notebook or script, and import it",
    )
  return None


def _payload(engine: CatEngine) -> bytes:
  """Return the engine pickled for the workers, checking that they could import every class and function it refers to.

  Each of the engine's fields is checked on its own first, so that the error names the strategy at
  fault, then the engine itself, whose class may be a subclass of :class:`~catsim.CatEngine`.

  Parameters
  ----------
  engine : CatEngine
      The engine to send to the workers.

  Returns
  -------
  bytes
      The pickled engine.

  Raises
  ------
  ValueError
      If the engine refers to a class or function the workers could not import, or cannot be
      pickled.
  """
  for field in dataclasses.fields(engine):
    component = getattr(engine, field.name)
    _checked_pickle(f"the engine's {field.name} ({type(component).__name__})", component)
  return _checked_pickle(f"the engine ({type(engine).__name__})", engine)


def _checked_pickle(label: str, obj: object) -> bytes:
  """Return ``obj`` pickled, raising if the workers could not import a class or function it refers to.

  Parameters
  ----------
  label : str
      What ``obj`` is, for the error message.
  obj : object
      The object to pickle.

  Returns
  -------
  bytes
      The pickle.

  Raises
  ------
  ValueError
      If ``obj`` refers to a class or function the workers could not import, or cannot be pickled.
  """
  where = (
    "simulate runs the parts on worker processes, which receive the engine pickled and import every class and "
    f"function it refers to by its module and name; {label}"
  )
  buffer = io.BytesIO()
  try:
    _ImportCheck(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
  except _UnimportableError as error:
    msg = f"{where} refers to {error.description}, which they cannot import. {error.advice}, or run with workers=1"
    raise ValueError(msg) from None
  except (pickle.PicklingError, TypeError, AttributeError) as error:
    msg = f"{where} cannot be pickled ({error}). Run with workers=1, or keep only picklable values on it"
    raise ValueError(msg) from error
  return buffer.getvalue()


def _run_on_workers[Task, Result](
  function: Callable[[CatEngine, Task, Callable[[], object] | None], Result],
  payload: bytes,
  tasks: Sequence[Task],
  *,
  workers: int,
  done: Callable[[], object] | None,
) -> list[Result]:
  """Run ``function(engine, task, report)`` for every task on worker processes, and return the results in task order.

  Each task unpickles its own copy of the engine from ``payload``, so a task's result is what
  ``function`` gives on the engine in the calling process. The workers are spawned on every
  platform, as macOS and Windows do by default, so that the engines a run accepts do not depend on
  the platform: a forked worker would inherit what a notebook defined, and a run that worked on
  Linux would fail elsewhere. Spawning also avoids forking a process whose threads, such as NumPy's,
  may hold locks.

  Parameters
  ----------
  function : callable
      A module-level function, so that the workers can import it. In a worker, ``report`` is a
      function to call after each session ends, or None when ``done`` is None.
  payload : bytes
      The engine, pickled by :func:`_payload`.
  tasks : Sequence
      The tasks, each picklable.
  workers : int
      The number of worker processes, at least 2; on Windows, whose process pools take at most 61,
      at most 61 run.
  done : callable or None
      Called in the calling process, once for each call a worker makes to ``report``, as the
      calling process hears of it. None to not report sessions.

  Returns
  -------
  list
      Each task's result, in the order of ``tasks``.

  Raises
  ------
  Exception
      A task's exception or the exception of ``done``, as soon as the calling process meets it, or
      ``concurrent.futures.process.BrokenProcessPool`` if a worker process died.
  """
  if sys.platform == "win32":
    workers = min(workers, _MAX_WINDOWS_WORKERS)
  context = multiprocessing.get_context("spawn")
  messages: Queue[_Message] | None = None if done is None else context.Queue()
  pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_listen, initargs=(messages,))
  try:
    futures = [pool.submit(_work, function, payload, task) for task in tasks]
    if messages is not None and done is not None:
      _relay(messages, futures, done)
    # The results cross from the workers as pickles, which carry no static type.
    results = [cast("Result", future.result()) for future in futures]
  except BaseException:
    # Raise now rather than once the running tasks end, which they do in the background; the tasks
    # not yet started never run.
    pool.shutdown(wait=False, cancel_futures=True)
    raise
  pool.shutdown()
  return results


def _listen(messages: Queue[_Message] | None) -> None:
  """Keep the run's message queue in a worker process, for the tasks it runs to report their sessions on.

  Parameters
  ----------
  messages : multiprocessing.Queue or None
      The queue the calling process reads, or None when it reads none.
  """
  if messages is not None:
    # A worker exits without waiting for its messages to be read: after a failure the calling process
    # stops reading, and a worker blocked on a full pipe would never exit.
    messages.cancel_join_thread()
  _worker["messages"] = messages


def _work(
  function: Callable[[CatEngine, Any, Callable[[], object] | None], object], payload: bytes, task: object
) -> object:
  """Run one task in a worker process: unpickle the engine and call the task's function, then say it has ended.

  The engine is unpickled here rather than by the pool, so that an import that fails in the worker
  reaches the calling process as the task's exception rather than as a broken pool.

  Parameters
  ----------
  function : callable
      The function :func:`_run_on_workers` was given.
  payload : bytes
      The pickled engine.
  task : object
      The task.

  Returns
  -------
  object
      What ``function`` returns.
  """
  messages = _worker["messages"]
  try:
    engine = pickle.loads(payload)  # noqa: S301 - pickled by the calling process, from its own engine
    report = None if messages is None else functools.partial(messages.put, _Message.SESSION)
    return function(engine, task, report)
  finally:
    if messages is not None:
      messages.put(_Message.TASK)


def _relay(messages: Queue[_Message], futures: list[Future[object]], done: Callable[[], object]) -> None:
  """Call ``done`` for every session the workers report, until every task has said it has ended.

  A worker sends its messages in order, so a task's sessions all arrive before its end. A worker
  drops messages only when it exits, which it does after the pool shuts down or by dying, and a
  worker's death breaks the pool and fails every task, which the next quiet poll raises.

  Parameters
  ----------
  messages : multiprocessing.Queue
      The queue the workers report on.
  futures : list of concurrent.futures.Future
      The tasks' futures.
  done : callable
      Called once for every session reported.
  """
  running = len(futures)
  while running:
    try:
      message = messages.get(timeout=_POLL_SECONDS)
    except queue.Empty:
      _raise_failure(futures)
      continue
    if message == _Message.SESSION:
      done()
    else:
      running -= 1
      _raise_failure(futures)


def _raise_failure(futures: list[Future[object]]) -> None:
  """Raise the exception of the first finished task that failed, if any.

  Parameters
  ----------
  futures : list of concurrent.futures.Future
      The tasks' futures.
  """
  for future in futures:
    if future.done():
      future.result()
