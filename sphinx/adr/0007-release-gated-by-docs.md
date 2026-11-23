# ADR-0007: Gate Package Publication on Documentation Availability

## Status

Accepted

## Context

For this project, a release is considered ready only when the corresponding documentation is
available online, not merely when the package builds and tests pass.

## Decision

Keep package publication gated by the documentation deployment path:

- TestPyPI publication depends on the dev documentation deployment
- PyPI publication depends on the production documentation deployment

## Consequences

- CI models documentation availability as part of release readiness.
- Publishing is intentionally coupled to the docs deployment DAG.
- The docs deployment and the publish it gates run in the same workflow run, because a job
  cannot wait for a job in another run: the GitHub release event deploys the production docs and
  then publishes to PyPI, and the `dev` → `main` pull request deploys the dev docs and then
  publishes a release candidate to TestPyPI. These are the only two publishing paths; there is no
  manual publish.
- The production docs go live before the PyPI upload waits for the maintainer's approval in the
  `pypi` environment, so a rejected upload leaves docs online for a version that is not on PyPI
  until the next deploy.
- Release automation stays aligned with the project's documentation policy.
