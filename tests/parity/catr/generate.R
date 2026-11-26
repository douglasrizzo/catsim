# Writes the catR outputs that tests/test_parity_catr.py compares catsim with (ADR-0020).
#
# Run from anywhere, with R installed:
#
#   Rscript tests/parity/catr/generate.R
#
# catR is loaded from the library named by the CATSIM_CATR_LIB environment variable, by default
# ~/.cache/catsim-parity/R-lib, outside the repository and the user's R library. When the pinned
# catR version is missing there, the script installs it from CRAN's source archive. Only base R and
# catR are used.
#
# Each case script builds the cases of one JSON file beside it: the inputs, which the Python test
# reads too, and catR's outputs, at the three levels of ADR-0020:
# 1. single computations: irt.R, estimation.R, selection.R and stopping.R;
# 2. whole tests on recorded responses: whole_tests.R, on the banks of selection.R and
#    simulations.R, which are sourced first;
# 3. simulations of the randomized selection rules: simulations.R, which takes about two minutes.
# Item indices in the files are 0-based, as in catsim; the scripts add 1 for catR and subtract 1
# from its answers.

CATR_VERSION <- "3.17"

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

install_catr <- function(lib) {
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  tarball <- sprintf("catR_%s.tar.gz", CATR_VERSION)
  # the current release sits in src/contrib, older ones in its Archive
  for (url in c(
    paste0("https://cloud.r-project.org/src/contrib/", tarball),
    paste0("https://cloud.r-project.org/src/contrib/Archive/catR/", tarball)
  )) {
    installed <- tryCatch(
      {
        install.packages(url, lib = lib, repos = NULL, type = "source", quiet = TRUE)
        TRUE
      },
      warning = function(w) FALSE,
      error = function(e) FALSE
    )
    if (installed) {
      return(invisible())
    }
  }
  stop("could not install catR ", CATR_VERSION, " into ", lib)
}

lib <- Sys.getenv("CATSIM_CATR_LIB", file.path(path.expand("~"), ".cache", "catsim-parity", "R-lib"))
has_pinned <- function() {
  requireNamespace("catR", lib.loc = lib, quietly = TRUE) &&
    as.character(packageVersion("catR", lib.loc = lib)) == CATR_VERSION
}
if (!has_pinned()) {
  install_catr(lib)
}
stopifnot(has_pinned())
suppressPackageStartupMessages(library(catR, lib.loc = lib))

here <- script_dir()
for (helper in c("json.R", "irt.R", "estimation.R", "selection.R", "stopping.R", "simulations.R", "whole_tests.R")) {
  source(file.path(here, helper))
}

versions <- list(catr_version = CATR_VERSION, r_version = R.version.string)
outputs <- list(
  irt = irt_cases(),
  estimation = estimation_cases(),
  selection = selection_cases(),
  stopping = stopping_cases(),
  whole_tests = whole_test_cases(),
  simulations = simulation_cases()
)
for (name in names(outputs)) {
  path <- file.path(here, paste0(name, ".json"))
  write_json(c(versions, outputs[[name]]), path)
  message("wrote ", path)
}
