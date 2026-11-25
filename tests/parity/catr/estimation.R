# Ability estimates (catR's thetaEst) and their standard errors (semTheta) on fixed response patterns.
#
# Every case names catR's method and settings; the Python test builds the catsim estimator that
# matches them. "se_slope" is the derivative of catR's standard error in the ability, a central
# difference of semTheta: the test allows for it when the two estimates differ by their tolerance.

ESTIMATION_BANK <- rbind(
  c(1.2, -1.5, 0, 1),
  c(0.8, -0.6, 0, 1),
  c(1.6, 0.0, 0, 1),
  c(1.0, 0.7, 0.2, 1),
  c(2.0, 1.3, 0.15, 1),
  c(1.4, -0.3, 0.1, 0.95),
  c(0.9, 1.9, 0.25, 0.9),
  c(1.7, 0.4, 0, 0.92)
)

PATTERNS <- list(
  one_correct = list(items = 2, responses = 1),
  one_incorrect = list(items = 2, responses = 0),
  three_correct = list(items = c(0, 1, 2), responses = c(1, 1, 1)),
  three_incorrect = list(items = c(3, 4, 5), responses = c(0, 0, 0)),
  two_mixed = list(items = c(0, 2), responses = c(1, 0)),
  four_mixed = list(items = c(1, 2, 3, 5), responses = c(1, 0, 1, 0)),
  five_mixed = list(items = c(0, 1, 2, 3, 5), responses = c(1, 1, 1, 0, 1)),
  eight_mixed = list(items = 0:7, responses = c(1, 1, 0, 1, 0, 1, 0, 0)),
  # a slip on an easy 4PL item and guessing: the likelihood has a maximum near -3 and a minimum near
  # 0 inside [-4, 4], and still rises at the upper bound
  rising_at_bound = list(items = c(5, 4, 6), responses = c(0, 1, 1)),
  # guessing on the two hard items: the likelihood has a minimum and a maximum inside [-4, 4]
  two_stationary_points = list(items = c(1, 4, 6), responses = c(0, 1, 1)),
  # guessing: Warm's modified score has three roots
  three_wl_roots = list(items = c(0, 4), responses = c(1, 0))
)

CONSTANT <- c("one_correct", "one_incorrect", "three_correct", "three_incorrect")
MIXED <- c("two_mixed", "four_mixed", "five_mixed", "eight_mixed", "rising_at_bound")

STANDARD_NORMAL <- list(distribution = "norm", parameters = c(0, 1))
SHIFTED_NORMAL <- list(distribution = "norm", parameters = c(0.5, 1.5))
UNIFORM <- list(distribution = "unif", parameters = c(-3, 3))

# One entry per catR configuration: the method, its settings and the patterns it runs on.
ESTIMATORS <- list(
  list(name = "ml", method = "ML", range = c(-4, 4), patterns = c(MIXED, "two_stationary_points")),
  # Dodd's step for constant patterns: catR's constantPatt = "var", from current.th toward the
  # bank's extreme difficulties
  list(name = "ml_dodd", method = "ML", range = c(-4, 4), constant_pattern = "var", current_theta = 0, patterns = "one_correct"),
  list(name = "ml_dodd", method = "ML", range = c(-4, 4), constant_pattern = "var", current_theta = -0.2, patterns = "three_incorrect"),
  list(name = "ml_dodd", method = "ML", range = c(-4, 4), constant_pattern = "var", current_theta = 0.95, patterns = "three_correct"),
  list(name = "ml_dodd_past_hardest", method = "ML", range = c(-4, 4), constant_pattern = "var", current_theta = 2.5, patterns = "three_correct"),
  # catR's default for constant patterns, constantPatt = NULL: the bound of the range
  list(name = "ml_constant_default", method = "ML", range = c(-4, 4), patterns = c("three_correct", "three_incorrect")),
  list(name = "bm", method = "BM", range = c(-4, 4), prior = STANDARD_NORMAL, patterns = c(CONSTANT, MIXED)),
  list(name = "bm_shifted_prior", method = "BM", range = c(-4, 4), prior = SHIFTED_NORMAL, patterns = c("one_incorrect", "three_correct", "four_mixed", "eight_mixed")),
  # catR's Bayes modal estimate with a uniform prior is the ML estimate on the prior's support
  list(name = "bm_uniform_prior", method = "BM", range = UNIFORM$parameters, prior = UNIFORM, patterns = c("three_correct", "three_incorrect", "four_mixed", "eight_mixed")),
  list(name = "eap", method = "EAP", grid = c(-4, 4, 33), prior = STANDARD_NORMAL, patterns = c(CONSTANT, MIXED)),
  list(name = "eap_catsim_grid", method = "EAP", grid = c(-6, 6, 121), prior = SHIFTED_NORMAL, patterns = c("one_incorrect", "three_correct", "four_mixed", "eight_mixed")),
  list(name = "eap_uniform_prior", method = "EAP", grid = c(-4, 4, 33), prior = UNIFORM, patterns = c("three_correct", "four_mixed", "eight_mixed")),
  list(name = "wl", method = "WL", range = c(-4, 4), patterns = c(CONSTANT, MIXED, "three_wl_roots"))
)

estimate_case <- function(estimator, pattern_name) {
  pattern <- PATTERNS[[pattern_name]]
  items <- rbind(ESTIMATION_BANK[pattern$items + 1, ])
  x <- pattern$responses
  prior <- if (is.null(estimator$prior)) STANDARD_NORMAL else estimator$prior
  search_range <- if (is.null(estimator$range)) c(-4, 4) else estimator$range
  grid <- if (is.null(estimator$grid)) c(-4, 4, 33) else estimator$grid
  current_theta <- if (is.null(estimator$current_theta)) 0 else estimator$current_theta
  estimate <- function() {
    thetaEst(items, x,
      method = estimator$method, priorDist = prior$distribution, priorPar = prior$parameters,
      range = search_range, parInt = grid, constantPatt = estimator$constant_pattern,
      current.th = current_theta, bRange = range(ESTIMATION_BANK[, 2])
    )
  }
  standard_error <- function(theta) {
    semTheta(theta, items, x,
      method = estimator$method, priorDist = prior$distribution, priorPar = prior$parameters,
      parInt = grid, constantPatt = estimator$constant_pattern
    )
  }
  theta <- estimate()
  se <- standard_error(theta)
  step <- 1e-5
  slope <- if (is.finite(se)) (standard_error(theta + step) - standard_error(theta - step)) / (2 * step) else 0
  case <- list(
    id = paste0(estimator$name, "/", pattern_name),
    method = estimator$method,
    items = as_array(as.integer(pattern$items)),
    responses = as_array(x == 1),
    current_theta = current_theta,
    theta = theta,
    se = se,
    se_slope = slope
  )
  if (estimator$method %in% c("ML", "BM", "WL")) case$range <- search_range
  if (estimator$method %in% c("BM", "EAP")) case$prior <- list(distribution = prior$distribution, parameters = prior$parameters)
  if (estimator$method == "EAP") case$grid <- list(low = grid[1], high = grid[2], nodes = as.integer(grid[3]))
  if (!is.null(estimator$constant_pattern)) case$constant_pattern <- estimator$constant_pattern
  case
}

estimation_cases <- function() {
  cases <- list()
  for (estimator in ESTIMATORS) {
    for (pattern_name in estimator$patterns) {
      cases[[length(cases) + 1L]] <- estimate_case(estimator, pattern_name)
    }
  }
  list(bank = ESTIMATION_BANK, cases = cases)
}
