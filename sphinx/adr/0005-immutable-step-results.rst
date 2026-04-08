ADR-0005: Make Step Results Immutable Snapshots
===============================================

Status
------

Accepted

Context
-------

Returning a live mutable session object from step results made earlier results drift as the session
continued to evolve. That weakened the semantics of step-level inspection, tracing, and tutorials.

Decision
--------

Make ``CatStepResult.session`` an immutable snapshot of session state at the end of the step rather
than a reference to the live mutable session object.

Consequences
------------

* Stored step results remain stable after later CAT steps.
* Step-level debugging and audit-style inspection become reliable.
* The engine continues to mutate ``CatSessionState`` internally, but result objects have durable
  semantics.
