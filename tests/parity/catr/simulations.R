# Simulations of the randomized selection rules (catR's simulateRespondents), level 3 of ADR-0020.
#
# R and NumPy draw different numbers, so these cases compare summary statistics within their Monte
# Carlo error rather than single tests. Each case simulates the examinees whose true abilities are
# TRUE_THETAS, the quantiles (i - 0.5) / n of the standard normal distribution, on
# SIMULATION_BANK, with catR's own response generation. The start has no early item (nrItems = 0)
# at ability 0, as in whole_tests.R, and the provisional and final estimators are both EAP on
# catR's grid with a standard normal prior, so the final estimate is the last provisional one, as
# in catsim.
#
# Each case records catR's statistics with their standard errors over the n examinees, which treat
# the abilities as drawn at random. Both packages use the same fixed abilities, so these standard
# errors, which count the spread of a statistic between abilities as noise, overstate the Monte
# Carlo error of the difference and make the comparison conservative (test_parity_catr.py's
# TestSimulations gives an example):
# - bias: the mean error (final estimate minus true ability), with sd(errors) / sqrt(n);
# - RMSE: the square root of the mean squared error, with the standard error that the delta method
#   gives from the MSE's, sd(squared errors) / (2 RMSE sqrt(n)) (Morris, White and Crowther, 2019);
# - mean test length, with sd(lengths) / sqrt(n);
# - overlap rate: catR's sum of squared exposure rates over the mean test length, with the
#   jackknife standard error over examinees;
# - every item's exposure count.
# The script checks that these are the statistics simulateRespondents reports.
#
# catR re-seeds R's generator from the clock (set.seed(NULL)) at the end of nextItem, startItems
# and genPattern, so its simulations cannot be repeated. The cases run copies of those functions,
# and of randomCAT and simulateRespondents, which call them, whose set.seed ignores NULL: each case
# then draws from one stream, seeded at its start. The draws are as random as catR's; only their
# source no longer changes from run to run.

SIMULATION_BANK <- local({
  set.seed(20261126L)
  n_items <- 50L
  bank <- cbind(round(runif(n_items, 0.8, 2), 2), round(rnorm(n_items), 3), round(runif(n_items, 0, 0.2), 2), 1)
  # bOpt in whole_tests.R would tie on equal difficulties
  stopifnot(!anyDuplicated(bank[, 2]))
  bank
})

SIMULATION_EXAMINEES <- 1000L
TRUE_THETAS <- qnorm((seq_len(SIMULATION_EXAMINEES) - 0.5) / SIMULATION_EXAMINEES)
EAP_SETTINGS <- list(method = "EAP", priorDist = "norm", priorPar = c(0, 1), parInt = GRID)

# The randomized rules: randomesque (catR's randomesque with MFI), progressive and proportional,
# with catR's default acceleration AP = 1 and the schedule over the test's length. catR has no
# 5-4-3-2-1 rule.
SIMULATIONS <- list(
  list(id = "randomesque", test = list(itemSelect = "MFI", randomesque = 5L), stop = list(rule = "length", thr = 10)),
  list(
    id = "randomesque_precision", test = list(itemSelect = "MFI", randomesque = 5L),
    stop = list(rule = c("precision", "length"), thr = c(0.4, 20))
  ),
  list(
    id = "progressive", test = list(itemSelect = "progressive", AP = 1, proRule = "length", proThr = 10),
    stop = list(rule = "length", thr = 10)
  ),
  list(
    id = "proportional", test = list(itemSelect = "proportional", AP = 1, proRule = "length", proThr = 10),
    stop = list(rule = "length", thr = 10)
  )
)

# Copies of catR's functions that call set.seed(NULL), and of those that call them, in an
# environment whose set.seed ignores NULL; every other function is catR's own
reseeding_off <- function() {
  env <- new.env(parent = asNamespace("catR"))
  env$set.seed <- function(seed, ...) if (!is.null(seed)) base::set.seed(seed, ...)
  for (name in c("simulateRespondents", "randomCAT", "startItems", "nextItem", "genPattern")) {
    f <- get(name, envir = asNamespace("catR"))
    environment(f) <- env
    assign(name, f, envir = env)
  }
  env
}

simulation_summary <- function(result) {
  n <- length(TRUE_THETAS)
  errors <- result$estimatedThetas - TRUE_THETAS
  rmse <- sqrt(mean(errors^2))
  given <- as.matrix(result$responses.df[, grep("^items\\.administrated", names(result$responses.df))])
  # one row per examinee, 1 for each item administered; unused columns hold -99
  administered <- t(apply(given, 1, function(row) tabulate(row[row > 0], nbins = nrow(SIMULATION_BANK))))
  stopifnot(all(administered <= 1))
  counts <- colSums(administered)
  lengths <- rowSums(administered)
  rates <- counts / n
  overlap <- sum(rates^2) / sum(rates)
  left_out <- sweep(-administered, 2, counts, "+") / (n - 1)
  overlaps <- rowSums(left_out^2) / rowSums(left_out)
  stopifnot(
    lengths == result$numberItems,
    isTRUE(all.equal(mean(errors), result$bias)),
    isTRUE(all.equal(rmse, result$RMSE)),
    isTRUE(all.equal(mean(lengths), result$testLength)),
    isTRUE(all.equal(rates, result$exposureRates)),
    isTRUE(all.equal(overlap, result$overlap))
  )
  list(
    bias = mean(errors),
    bias_se = sd(errors) / sqrt(n),
    rmse = rmse,
    rmse_se = sd(errors^2) / (2 * rmse * sqrt(n)),
    mean_length = mean(lengths),
    mean_length_se = sd(lengths) / sqrt(n),
    overlap = overlap,
    overlap_se = sqrt((n - 1) / n * sum((overlaps - mean(overlaps))^2)),
    exposure_counts = as_array(as.integer(counts))
  )
}

simulation_cases <- function() {
  catr <- reseeding_off()
  cases <- lapply(seq_along(SIMULATIONS), function(k) {
    simulation <- SIMULATIONS[[k]]
    set.seed(20261125L + k)
    # simulateRespondents reports its progress with cat()
    utils::capture.output(result <- catr$simulateRespondents(TRUE_THETAS, SIMULATION_BANK,
      start = list(nrItems = 0L, theta = 0), test = c(EAP_SETTINGS, simulation$test),
      stop = simulation$stop, final = EAP_SETTINGS
    ))
    settings <- simulation$test
    c(
      list(
        id = simulation$id,
        criterion = settings$itemSelect,
        rules = as_array(simulation$stop$rule),
        thresholds = as_array(simulation$stop$thr)
      ),
      if (settings$itemSelect == "MFI") list(randomesque = settings$randomesque),
      if (settings$itemSelect != "MFI") list(acceleration = settings$AP, test_size = as.integer(settings$proThr)),
      simulation_summary(result)
    )
  })
  list(
    bank = SIMULATION_BANK,
    true_thetas = as_array(TRUE_THETAS),
    prior = list(distribution = EAP_SETTINGS$priorDist, parameters = EAP_SETTINGS$priorPar),
    grid = list(low = GRID[1], high = GRID[2], nodes = as.integer(GRID[3])),
    cases = cases
  )
}
