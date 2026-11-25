"""Reference outputs: catsim's deterministic computations as recorded from dev at adda72dc.

The rescoping of catsim rewrites its core API and much of its test suite. These outputs prove that
no computation is lost on the way, independently of the tests that the rewrite changes:
``tests/test_reference_outputs.py`` recomputes every case of :mod:`tests.reference.cases` with the
current code and compares the result with the data in ``tests/reference/data/``, exactly for
integers, Booleans and strings and to a relative 1e-12 for floats.

Rules for later changes:

- The data changes only where an ADR changes a result. The pull request that changes it names the
  ADR and the outputs it changes, and regenerates the data with ``python -m tests.reference.generate``.
- Every recorded output is expected to survive every item of the rescoping unchanged. Simulations
  with generated examinees are left out on purpose, since ADR-0013 changes their random model; the
  whole tests here answer from recorded responses with configurations that draw no random number.
- A change of calling convention, such as the immutable ``Session`` or ``simulate``, rewrites
  :mod:`tests.reference.adapters` only: every call into the step, session and simulation API goes
  through it, and the data stays as it is.
"""
