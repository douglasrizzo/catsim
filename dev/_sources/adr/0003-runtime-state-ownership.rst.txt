ADR-0003: Keep Runtime State Outside Domain Objects
===================================================

Status
------

Accepted

Context
-------

Exposure tracking and per-session execution details were previously entangled with long-lived
objects in ways that made repeated runs and object reuse harder to reason about.

Decision
--------

Treat calibrated item data as domain data and keep runtime state in dedicated execution models:

* ``ItemBank`` holds item parameters and related derived data
* ``CatSessionState`` holds per-examinee runtime state
* ``ExposureTracker`` and ``SimulationResult`` hold run-level exposure information

Consequences
------------

* Reusing an item bank across sessions or simulations is safer.
* Exposure becomes a property of a run, not of the calibrated bank itself.
* Run metrics have a clearer ownership model and are easier to validate.
