# Stopping decisions (catR's checkStopRule) for the length, precision and classification rules.
#
# Each case gives the number of administered items, the estimate and its standard error, and one or
# more rules with their thresholds. "decision" is checkStopRule's for all the rules together, and
# "per_rule" its decision for each rule alone.

z_score <- function(alpha) qnorm(1 - alpha / 2)

STOPPING_CASES <- list(
  list(id = "length/below", rules = "length", thresholds = 10, n_items = 9L, theta = 0.2, se = 0.4),
  list(id = "length/at", rules = "length", thresholds = 10, n_items = 10L, theta = 0.2, se = 0.4),
  list(id = "length/above", rules = "length", thresholds = 10, n_items = 11L, theta = 0.2, se = 0.4),
  list(id = "precision/above", rules = "precision", thresholds = 0.3, n_items = 6L, theta = 0.2, se = 0.3000001),
  list(id = "precision/at", rules = "precision", thresholds = 0.3, n_items = 6L, theta = 0.2, se = 0.3),
  list(id = "precision/below", rules = "precision", thresholds = 0.3, n_items = 6L, theta = 0.2, se = 0.2999),
  list(id = "precision/no_response", rules = "precision", thresholds = 0.3, n_items = 0L, theta = 0, se = Inf),
  list(id = "classification/above_threshold", rules = "classification", thresholds = 0.5, n_items = 8L, theta = 1.2, se = 0.3),
  list(id = "classification/straddles", rules = "classification", thresholds = 0.5, n_items = 8L, theta = 0.8, se = 0.3),
  list(id = "classification/below_threshold", rules = "classification", thresholds = 0, n_items = 8L, theta = -0.4, se = 0.2),
  list(id = "classification/alpha_0.1", rules = "classification", thresholds = 0, n_items = 8L, theta = -0.4, se = 0.24, alpha = 0.1),
  list(id = "classification/alpha_0.01", rules = "classification", thresholds = 0, n_items = 8L, theta = -0.4, se = 0.24, alpha = 0.01),
  list(id = "classification/no_response", rules = "classification", thresholds = 0, n_items = 0L, theta = 0, se = Inf),
  # the interval's lower end 1e-9 above and below the threshold
  list(id = "classification/just_clears", rules = "classification", thresholds = 0.5, n_items = 8L, theta = 0.5 + z_score(0.05) * 0.3 + 1e-9, se = 0.3),
  list(id = "classification/just_straddles", rules = "classification", thresholds = 0.5, n_items = 8L, theta = 0.5 + z_score(0.05) * 0.3 - 1e-9, se = 0.3),
  list(id = "length_or_precision/precision", rules = c("length", "precision"), thresholds = c(10, 0.3), n_items = 7L, theta = 0.2, se = 0.25),
  list(id = "length_or_precision/length", rules = c("length", "precision"), thresholds = c(10, 0.3), n_items = 10L, theta = 0.2, se = 0.5),
  list(id = "length_or_precision/neither", rules = c("length", "precision"), thresholds = c(10, 0.3), n_items = 7L, theta = 0.2, se = 0.5),
  # two classification thresholds: catR stops once the interval clears either of them
  list(id = "two_thresholds/clears_both", rules = c("classification", "classification"), thresholds = c(-1, 1), n_items = 8L, theta = 0.2, se = 0.3),
  list(id = "two_thresholds/clears_one", rules = c("classification", "classification"), thresholds = c(-1, 1), n_items = 8L, theta = 0.9, se = 0.3),
  list(id = "two_thresholds/clears_neither", rules = c("classification", "classification"), thresholds = c(-1, 1), n_items = 8L, theta = 0, se = 0.6)
)

stopping_cases <- function() {
  cases <- lapply(STOPPING_CASES, function(case) {
    alpha <- if (is.null(case$alpha)) 0.05 else case$alpha
    check <- function(rules, thresholds) {
      checkStopRule(case$theta, case$se, case$n_items, stop = list(rule = rules, thr = thresholds, alpha = alpha))$decision
    }
    c(case[c("id", "n_items", "theta", "se")], list(
      rules = as_array(case$rules),
      thresholds = as_array(case$thresholds),
      alpha = alpha,
      decision = check(case$rules, case$thresholds),
      per_rule = as_array(mapply(check, case$rules, case$thresholds))
    ))
  })
  list(cases = cases)
}
