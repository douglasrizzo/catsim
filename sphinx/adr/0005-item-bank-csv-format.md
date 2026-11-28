# ADR-0005: Item Banks Read and Write One CSV Format

## Status

Accepted 2026-11-25; superseded 2026-11-27 by irtlib's ADR-0009, the item bank format

## Context

catsim's item bank read and wrote one CSV format: a header row, the parameter columns `a` to `d` by
name with the defaults of ADR-0003 for the missing ones, an optional name column, and nothing else.
The item bank is irtlib's (ADR-0024), and so is its file format.

## Decision

Item banks are read and written by irtlib's `ItemBank.from_csv` and `ItemBank.to_csv`, in the
format irtlib's ADR-0009 sets out. catsim has no reader or writer of its own.

## Consequences

- A bank file that a study saves loads in catsim and in irtlib alike, since there is one reader.
- A change to the format is made, and recorded, in irtlib.

## Alternatives Considered

- Keeping this record of the format beside irtlib's: two records of one decision drift.
