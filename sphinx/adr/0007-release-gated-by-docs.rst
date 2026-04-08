ADR-0007: Gate Package Publication on Documentation Availability
================================================================

Status
------

Accepted

Context
-------

For this project, a release is considered ready only when the corresponding documentation is
available online, not merely when the package builds and tests pass.

Decision
--------

Keep package publication gated by the documentation deployment path:

* TestPyPI publication depends on the dev documentation deployment
* PyPI publication depends on the production documentation deployment

Consequences
------------

* CI models documentation availability as part of release readiness.
* Publishing is intentionally coupled to the docs deployment DAG.
* Release automation stays aligned with the project's documentation policy.
