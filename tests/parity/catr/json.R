# A JSON writer in base R, so the generator needs no package besides catR.
#
# The layout is Python's json.dumps with indent 2 and sorted keys. Numbers have 17 significant
# digits, which a correctly rounding parser, such as Python's, reads back as the same double. Fewer
# digits would do for most numbers, but R's own parser is not correctly rounded, so R cannot check
# which ones; the pre-commit hooks leave these files alone (pretty-format-json would shorten them).
# Non-finite numbers are written as the strings "Inf", "-Inf" and "NaN".
#
# A named list is an object and an unnamed list an array; an atomic vector of length 1 is a scalar
# and any other length an array, unless wrapped in as_array(); a matrix is an array of its rows.

as_array <- function(x) as.list(unname(x))

json_string <- function(s) {
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  paste0("\"", gsub("\"", "\\\"", s, fixed = TRUE), "\"")
}

json_number <- function(v) {
  if (!is.finite(v)) {
    return(json_string(format(v)))
  }
  if (v == 0) {
    return("0") # also for -0, which Python reads as the integer 0
  }
  sprintf("%.17g", v)
}

json_scalar <- function(v) {
  if (is.logical(v)) {
    return(if (v) "true" else "false")
  }
  if (is.character(v)) {
    return(json_string(v))
  }
  if (is.integer(v)) {
    return(sprintf("%d", v))
  }
  json_number(v)
}

to_json <- function(x, indent = 0L) {
  inner <- strrep("  ", indent + 1L)
  outer <- strrep("  ", indent)
  if (is.matrix(x)) {
    return(to_json(lapply(seq_len(nrow(x)), function(i) as_array(x[i, ])), indent))
  }
  if (is.atomic(x) && length(x) == 1L) {
    return(json_scalar(x))
  }
  if (is.atomic(x)) {
    x <- as.list(unname(x))
  }
  if (length(x) == 0L) {
    return(if (is.null(names(x))) "[]" else "{}")
  }
  if (!is.null(names(x))) {
    keys <- names(x)
    entries <- vapply(order(keys, method = "radix"), function(i) {
      paste0(inner, json_string(keys[i]), ": ", to_json(x[[i]], indent + 1L))
    }, "")
    return(paste0("{\n", paste(entries, collapse = ",\n"), "\n", outer, "}"))
  }
  entries <- vapply(x, function(e) paste0(inner, to_json(e, indent + 1L)), "")
  paste0("[\n", paste(entries, collapse = ",\n"), "\n", outer, "]")
}

write_json <- function(x, path) {
  writeLines(to_json(x), path, useBytes = TRUE)
}
