# Item selection (catR's nextItem) and the criterion values behind it, in fixed contexts.
#
# A context is the administered items, their responses and the ability estimate. "ranking" is the
# order in which nextItem picks every available item when each pick is made unavailable for the
# next (nAvailable), which leaves the administered items and responses unchanged; "ranking_values"
# is the criterion value nextItem reports with each pick. "values" are the criterion values of
# every candidate, from catR's own function for the criterion, where it has one.

SELECTION_BANK <- rbind(
  c(1.10, -1.80, 0.00, 1.00),
  c(0.75, -1.10, 0.00, 1.00),
  c(1.45, -0.55, 0.12, 1.00),
  c(1.90, -0.20, 0.00, 0.96),
  c(0.95, 0.05, 0.18, 1.00),
  c(1.30, 0.35, 0.00, 1.00),
  c(2.20, 0.80, 0.20, 0.93),
  c(0.85, 1.15, 0.00, 0.90),
  c(1.65, 1.60, 0.10, 1.00),
  c(1.20, 2.10, 0.25, 0.97),
  c(0.70, -2.40, 0.05, 1.00),
  c(1.55, -0.90, 0.00, 0.94)
)

CONTEXTS <- list(
  start = list(administered = integer(0), responses = numeric(0), theta = 0),
  two_items = list(administered = c(5L, 2L), responses = c(1, 0), theta = -0.25),
  five_items = list(administered = c(5L, 2L, 3L, 8L, 11L), responses = c(1, 0, 1, 0, 1), theta = 0.65),
  all_correct = list(administered = c(4L, 6L, 0L), responses = c(1, 1, 1), theta = 1.4)
)
# For KL only: the best item differs between the windows of 1, 2 and 3 administered items
KL_CONTEXTS <- list(
  window_sensitive = list(administered = c(5L, 2L), responses = c(1, 0), theta = 0.95)
)

# catR's settings for the criteria that estimate or integrate: catsim's MLWISelector(n_nodes = 33,
# bounds = (-4, 4)) and MEISelector(MAPEstimator(bounds = (-4, 4))) match them
GRID <- c(-4, 4, 33)
MEI_SETTINGS <- list(method = "BM", priorDist = "norm", priorPar = c(0, 1), range = c(-4, 4), infoType = "Fisher")
# Chang and Ying's window constant, catsim's KLSelector default
KL_WINDOW <- 3

candidates_of <- function(context) setdiff(seq_len(nrow(SELECTION_BANK)) - 1L, context$administered)

# catR's arguments for out and x: NULL before the first item
out_of <- function(context) if (length(context$administered)) context$administered + 1L else NULL
x_of <- function(context) if (length(context$responses)) context$responses else NULL
given_of <- function(context) SELECTION_BANK[out_of(context), , drop = FALSE]

next_item <- function(context, criterion, available = NULL, ...) {
  catR::nextItem(SELECTION_BANK,
    theta = context$theta, out = out_of(context), x = x_of(context), criterion = criterion,
    parInt = GRID, randomesque = 1, nAvailable = available, ...
  )
}

ranking_of <- function(context, criterion, ...) {
  available <- rep(1, nrow(SELECTION_BANK))
  items <- integer(0)
  values <- numeric(0)
  for (step in seq_along(candidates_of(context))) {
    pick <- next_item(context, criterion, available = available, ...)
    items <- c(items, pick$item - 1L)
    values <- c(values, pick$info)
    available[pick$item] <- 0
  }
  list(ranking = as_array(items), ranking_values = as_array(values))
}

# Chang and Ying's KL index over catsim's window theta +- c / sqrt(n), as catR's KL() computes it
# with the likelihood set to 1: KL() integrates by the trapezoid rule, refined here by Richardson
# extrapolation of two grids, which agrees with the next finer pair to a relative 1e-14
kl_window <- function(context, item, n = length(context$administered)) {
  half_width <- KL_WINDOW / sqrt(max(1, n))
  trapezoid <- function(nodes) {
    X <- seq(context$theta - half_width, context$theta + half_width, length.out = nodes)
    KL(SELECTION_BANK, item, x = NULL, it.given = NULL, theta = context$theta, X = X, lik = rep(1, nodes), nqp = nodes)
  }
  (4 * trapezoid(8193) - trapezoid(4097)) / 3
}

criterion_values <- function(criterion, context) {
  candidates <- candidates_of(context) + 1L
  switch(criterion,
    MFI = Ii(context$theta, SELECTION_BANK)$Ii[candidates],
    MLWI = sapply(candidates, function(i) {
      MWI(SELECTION_BANK, i, x_of(context), given_of(context), lower = GRID[1], upper = GRID[2], nqp = GRID[3])
    }),
    MEI = sapply(candidates, function(i) {
      do.call(MEI, c(list(SELECTION_BANK, i, x_of(context), context$theta, given_of(context)), MEI_SETTINGS))
    }),
    KL = sapply(candidates, function(i) kl_window(context, i)),
    NULL
  )
}

# The largest slope of each candidate's information on [-4, 4], from catR's derivative dIi: MEI's
# value moves by at most this much per unit change of its look-ahead estimates
information_slopes <- function(context) {
  thetas <- seq(-4, 4, by = 0.001)
  sapply(candidates_of(context) + 1L, function(i) max(abs(Ii(thetas, SELECTION_BANK[i, ])$dIi)))
}

# MLWI's integrand, catR's likelihood times information at each node of its grid, summed with equal
# weights: catsim's rule applied to catR's integrand (MWI itself integrates by the trapezoid rule)
mlwi_node_sums <- function(context) {
  sapply(candidates_of(context) + 1L, function(i) {
    catR::MWI(SELECTION_BANK, i, x_of(context), given_of(context), lower = GRID[1], upper = GRID[2], nqp = GRID[3])
    sum(captured$crit.value)
  })
}

# A ranking is only well defined, in either package, if no two candidates nearly tie
check_untied <- function(values) {
  stopifnot(min(diff(sort(values))) > 1e-6 * max(abs(values)))
}

# A case ranks the candidates by nextItem with `criterion` (with its extra settings), or, without
# one, by the values of catR's function for `values_of`
ranked_case <- function(name, context_name, criterion = NULL, values_of = NULL, extra = list()) {
  context <- c(CONTEXTS, KL_CONTEXTS)[[context_name]]
  case <- list(
    id = paste0(name, "/", context_name),
    administered = as_array(context$administered),
    responses = as_array(context$responses == 1),
    theta = context$theta,
    candidates = as_array(candidates_of(context))
  )
  if (!is.null(criterion)) {
    ranking <- do.call(ranking_of, c(list(context, criterion), extra))
    check_untied(unlist(ranking$ranking_values))
    case <- c(case, ranking)
  }
  if (!is.null(values_of)) {
    values <- criterion_values(values_of, context)
    check_untied(values)
    case$values <- as_array(values)
    if (is.null(criterion)) case$ranking <- as_array(candidates_of(context)[order(-values)])
    if (values_of == "MEI") case$value_slopes <- as_array(information_slopes(context))
    if (values_of == "MLWI") case$node_sums <- as_array(mlwi_node_sums(context))
    if (values_of == "KL") case$half_width <- KL_WINDOW / sqrt(max(1, length(context$administered)))
  }
  case
}

# The schedules of the progressive and proportional selectors (Barrada et al., 2008), within and
# beyond the planned length N = thr, with catR's acceleration AP
SCHEDULE_ORDER <- c(5L, 2L, 3L, 8L, 11L, 0L, 9L, 1L) # the administration order, 0-based
SCHEDULE_THETA <- 0.3
SCHEDULES <- list(
  list(test_size = 5L, accelerations = c(0, 1, 2.5, -0.5), positions = 1:5),
  list(test_size = 8L, accelerations = 1, positions = 1:8),
  list(test_size = 5L, accelerations = 1, positions = 6:7, name = "past_test_size"),
  list(test_size = 1L, accelerations = 1, positions = 1L, name = "one_item_test")
)

# trace() reads the locals of nextItem and MWI as they return: the weight wq, the selection
# probabilities, the randomesque bin and MWI's integrand. It changes the namespace's copies, which
# catR::nextItem and catR::MWI call, not the attached ones
captured <- new.env()
capture_locals <- quote(for (name in c("wq", "randomValues", "probSelect", "keep", "crit.value")) {
  if (exists(name, inherits = FALSE)) assign(name, get(name, inherits = FALSE), envir = captured)
})

schedule_case <- function(criterion, test_size, acceleration, position, seed, name) {
  administered <- SCHEDULE_ORDER[seq_len(position - 1L)]
  context <- list(administered = administered, responses = rep(1, length(administered)), theta = SCHEDULE_THETA)
  rm(list = ls(captured), envir = captured)
  set.seed(seed)
  uniforms <- runif(nrow(SELECTION_BANK))
  set.seed(seed)
  pick <- next_item(context, criterion, rule = "length", thr = test_size, AP = acceleration)
  candidates <- candidates_of(context)
  case <- list(
    id = sprintf("%s/%sN%d_AP%s_position%d", criterion, name, test_size, format(acceleration), position),
    test_size = test_size,
    acceleration = acceleration,
    administered = as_array(administered),
    responses = as_array(context$responses == 1),
    theta = SCHEDULE_THETA,
    candidates = as_array(candidates)
  )
  if (criterion == "progressive") {
    info <- Ii(SCHEDULE_THETA, SELECTION_BANK)$Ii
    # catR draws its random component as runif(n, 0, M): M times the uniforms drawn here
    stopifnot(identical(captured$randomValues, max(info[candidates + 1L]) * uniforms))
    case$weight <- captured$wq
    case$uniforms <- as_array(uniforms)
    case$item <- pick$item - 1L
  } else {
    case$exponent <- captured$wq
    case$probabilities <- as_array(captured$probSelect[candidates + 1L])
  }
  case
}

# The randomesque bin: the items nextItem draws from with criterion = "MFI" and randomesque = k,
# best first. The 5-4-3-2-1 method, which catR does not have, draws the item at position n from
# the 6 - n most informative items, catR's randomesque bin with k = max(1, 5 - administered items)
BIN_SIZES <- list(randomesque = function(context) 3L, the54321 = function(context) max(1L, 5L - length(context$administered)))

pool_case <- function(name, context_name) {
  context <- CONTEXTS[[context_name]]
  bin_size <- BIN_SIZES[[name]](context)
  rm(list = ls(captured), envir = captured)
  catR::nextItem(SELECTION_BANK,
    theta = context$theta, out = out_of(context), x = x_of(context), criterion = "MFI", randomesque = bin_size
  )
  list(
    id = paste0(name, "/", context_name),
    administered = as_array(context$administered),
    responses = as_array(context$responses == 1),
    theta = context$theta,
    bin_size = bin_size,
    pool = as_array(as.integer(captured$keep) - 1L)
  )
}

selection_cases <- function() {
  for (traced in c("nextItem", "MWI")) {
    suppressMessages(trace(traced, exit = capture_locals, print = FALSE, where = asNamespace("catR")))
  }
  on.exit(suppressMessages(for (traced in c("nextItem", "MWI")) untrace(traced, where = asNamespace("catR"))))
  cases <- list()
  add <- function(case) cases[[length(cases) + 1L]] <<- case
  for (context_name in names(CONTEXTS)) {
    add(ranked_case("mfi", context_name, criterion = "MFI", values_of = "MFI"))
    add(ranked_case("bopt", context_name, criterion = "bOpt"))
    add(ranked_case("mlwi", context_name, criterion = "MLWI", values_of = "MLWI"))
    add(ranked_case("kl_window", context_name, values_of = "KL"))
    for (name in names(BIN_SIZES)) add(pool_case(name, context_name))
  }
  sensitive <- KL_CONTEXTS$window_sensitive
  best <- sapply(1:3, function(n) {
    candidates <- candidates_of(sensitive)
    candidates[which.max(sapply(candidates + 1L, function(i) kl_window(sensitive, i, n)))]
  })
  stopifnot(!anyDuplicated(best))
  add(ranked_case("kl_window", "window_sensitive", values_of = "KL"))
  # catR's own KL rule, which weights the divergence by the likelihood over parInt
  add(ranked_case("kl", "two_items", criterion = "KL"))
  for (context_name in c("start", "two_items")) {
    add(ranked_case("mei", context_name, criterion = "MEI", values_of = "MEI", extra = MEI_SETTINGS))
  }
  seed <- 20261125L
  for (criterion in c("progressive", "proportional")) {
    for (schedule in SCHEDULES) {
      name <- if (is.null(schedule$name)) "" else paste0(schedule$name, "_")
      for (acceleration in schedule$accelerations) {
        for (position in schedule$positions) {
          seed <- seed + 1L
          add(schedule_case(criterion, schedule$test_size, acceleration, position, seed, name))
        }
      }
    }
  }
  list(bank = SELECTION_BANK, cases = cases)
}
