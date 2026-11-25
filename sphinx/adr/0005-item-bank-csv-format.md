# ADR-0005: Item Banks Read and Write One CSV Format

## Status

Accepted 2026-11-25

## Context

An item bank could only be built from a NumPy matrix, so every study began by parsing its bank
file by hand, and a generated bank could not be saved to repeat or share a study. Researchers asked
to load a bank from a CSV file. catR has no reader: its users call `read.csv` and pass the matrix,
whose row names become the item names, and save a bank with `write.csv`.

## Decision

- `ItemBank.from_csv(path)` reads a bank, and `bank.to_csv(path)` writes one, in one format:
  - comma-separated, with a dot as the decimal mark;
  - a header row, which is required;
  - the parameter columns named `a`, `b`, `c` and `d`, in any order. `b` is required; a missing `a`
    is 1, a missing `c` is 0 and a missing `d` is 1, the rule of ADR-0003 applied by column name;
  - an optional name column, headed `name` or with an empty header, the header R's `write.csv`
    gives row names. Without it, names are generated (ADR-0004).
- Any other column is an error that names it.
- `to_csv` writes the name column and the four parameter columns, with each number written so that
  it reads back exactly, so `ItemBank.from_csv(path)` gives back the bank that wrote it.
- The reader uses Python's `csv` module, and the bank validates what it reads as it validates a
  matrix.

## Consequences

- A bank saved from R with `write.csv(bank)` loads with its names, when its columns are named `a` to
  `d`.
- A generated bank can be saved with a study and shared with the R scripts of the parity tests
  (ADR-0020).
- A misspelled column, `A` or `disc`, raises an error instead of silently falling back to a default.
- Files in another layout, with other column names, other delimiters or a decimal comma, are read
  with pandas or NumPy and passed as `ItemBank(matrix, names=...)`.
- Reading a CSV file adds no dependency.

## Alternatives Considered

- Columns by position, as in the matrix: a file with an extra or reordered column would load
  silently wrong.
- Ignoring unknown columns, so a file can carry other metadata: a misspelled parameter column would
  be ignored too, and its parameter defaulted.
- Options for the delimiter, the decimal mark and the column names: pandas already reads every
  layout, and one format is what a saved study needs.
- A reader built on pandas: a dependency for one file format (ADR-0021).
