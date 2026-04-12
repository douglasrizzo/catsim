# ADR-0006: Version Tutorial Notebooks as .ipynb Files

## Status

Accepted

## Context

The tutorial material needs to live alongside the package, open easily in Colab, and remain
executable as the API evolves.

## Decision

Keep tutorial notebooks as version-controlled `.ipynb` files under `notebooks/` and validate
them through the repository toolchain.

## Consequences

- Users get stable GitHub and Colab entry points for tutorials.
- Notebook outputs are normalized through notebook-specific pre-commit hooks.
- Notebook execution is treated as validation of documentation, not as optional polish.
