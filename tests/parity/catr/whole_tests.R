# Whole tests on recorded responses (catR's randomCAT with `responses`), level 2 of ADR-0020.
#
# A case is one recorded examinee's test under one configuration without randomization: a selection
# rule (MFI, bOpt or MLWI, on catR's grid parInt = c(-4, 4, 33)), an estimator (ML, BM, EAP or WL,
# with the range and prior of estimation.R) and catR's precision and length rules, which stop the
# test at whichever holds first. catR starts with no early item (nrItems = 0) at ability 0, so the
# selection rule chooses the first item at that ability with no responses, as catsim's
# FixedPointInitializer(0) and selector do. catR's default start, one early item that startItems
# chooses at ability 0 by the test's own criterion (startSelect = "MFI" or "bOpt"), is checked to
# give the same tests for the MFI and bOpt cases with ML and EAP on the small bank, so those cases
# stand for it too. ML takes constantPatt = "var", Dodd's step from the previous estimate toward
# the bank's range of difficulties, which catsim's MLEstimator always takes
# (ADR-0018); the other estimators keep catR's default, their own estimate for constant patterns.
#
# "thetas" are the estimates in order, the starting ability first, "ses" the standard error after
# each response and "se_slopes" its derivative in the ability, as in estimation.R. Every choice and
# stopping decision of a case is checked to be robust to the estimates' tolerance: the same item is
# chosen at an estimate moved by WHOLE_TEST_TOLERANCE either way, and neither catR's standard error
# nor catsim's lies within that tolerance, times its slope plus 1, of the precision threshold.
# They differ only where catR's is infinite, for ML on a constant pattern: catsim then reports the
# Fisher standard error at Dodd's estimate, which is semTheta's without constantPatt.

# thetaEst's uniroot tolerance, 2^-13, plus catsim's, 1e-6: test_parity_catr.py's _ESTIMATE_ATOL
WHOLE_TEST_TOLERANCE <- 2^-13 + 1e-6

WHOLE_TEST_ESTIMATORS <- list(
  ml = list(method = "ML", range = c(-4, 4), constantPatt = "var"),
  bm = list(method = "BM", range = c(-4, 4), priorDist = "norm", priorPar = c(0, 1)),
  eap = list(method = "EAP", priorDist = "norm", priorPar = c(0, 1)),
  wl = list(method = "WL", range = c(-4, 4))
)
WHOLE_TEST_CRITERIA <- c(mfi = "MFI", bopt = "bOpt", mlwi = "MLWI")

# The banks and their recorded examinees: response patterns drawn once at fixed abilities, and on
# the small bank an examinee who answers every item correctly, whose pattern stays constant
RECORDED_BANKS <- list(
  selection = list(
    bank = SELECTION_BANK, thetas = c(-1.5, -0.4, 0.4, 1.5), all_correct = TRUE, seed = 1L,
    stop = list(rule = c("precision", "length"), thr = c(0.55, 8))
  ),
  simulation = list(
    bank = SIMULATION_BANK, thetas = c(-1.2, 0, 1.2), all_correct = FALSE, seed = 9L,
    stop = list(rule = c("precision", "length"), thr = c(0.4, 15))
  )
)

recorded_examinees <- function(recorded) {
  set.seed(recorded$seed)
  patterns <- t(sapply(recorded$thetas, function(theta) {
    as.integer(runif(nrow(recorded$bank)) < Pi(theta, recorded$bank)$Pi)
  }))
  labels <- paste0("theta_", as.character(recorded$thetas))
  if (recorded$all_correct) {
    patterns <- rbind(patterns, 1L)
    labels <- c(labels, "all_correct")
  }
  list(patterns = patterns, labels = labels)
}

whole_test_case <- function(bank_name, examinees, examinee, criterion_name, estimator_name, start_items = FALSE) {
  recorded <- RECORDED_BANKS[[bank_name]]
  bank <- recorded$bank
  criterion <- WHOLE_TEST_CRITERIA[[criterion_name]]
  estimator <- modifyList(list(priorDist = "norm", priorPar = c(0, 1)), WHOLE_TEST_ESTIMATORS[[estimator_name]])
  start <- if (start_items) list(nrItems = 1L, theta = 0, startSelect = criterion) else list(nrItems = 0L, theta = 0)
  result <- randomCAT(
    itemBank = bank, responses = examinees$patterns[examinee, ], start = start,
    test = c(estimator, list(parInt = GRID, itemSelect = criterion)), stop = recorded$stop
  )
  items <- result$testItems
  x <- result$pattern
  # with an early item, randomCAT's estimates start after its response
  thetas <- if (start_items) c(0, result$thetaProv) else result$thetaProv
  ses <- if (start_items) result$seProv else result$seProv[-1]
  stopifnot(length(thetas) == length(items) + 1L, length(ses) == length(items))

  se_at <- function(k, theta, constant_pattern = estimator$constantPatt) {
    semTheta(theta, rbind(bank[items[seq_len(k)], ]), x[seq_len(k)],
      method = estimator$method, priorDist = estimator$priorDist, priorPar = estimator$priorPar,
      parInt = GRID, constantPatt = constant_pattern
    )
  }
  step <- 1e-5
  slope_at <- function(k, ...) (se_at(k, thetas[k + 1L] + step, ...) - se_at(k, thetas[k + 1L] - step, ...)) / (2 * step)
  slopes <- sapply(seq_along(items), function(k) {
    stopifnot(identical(se_at(k, thetas[k + 1L]), ses[k]))
    if (is.finite(ses[k])) slope_at(k) else 0
  })

  for (k in seq_along(items)) {
    given <- seq_len(k - 1L)
    for (shift in c(-1, 1) * WHOLE_TEST_TOLERANCE) {
      pick <- nextItem(bank,
        theta = thetas[k] + shift, out = if (k > 1L) items[given] else NULL,
        x = if (k > 1L) x[given] else NULL, criterion = criterion, parInt = GRID
      )
      stopifnot(pick$item == items[k])
    }
  }
  precision <- recorded$stop$thr[recorded$stop$rule == "precision"]
  catsim_ses <- ses
  catsim_slopes <- slopes
  for (k in which(!is.finite(ses))) {
    catsim_ses[k] <- se_at(k, thetas[k + 1L], constant_pattern = NULL)
    catsim_slopes[k] <- slope_at(k, constant_pattern = NULL)
  }
  stopifnot(abs(catsim_ses - precision) > WHOLE_TEST_TOLERANCE * (abs(catsim_slopes) + 1))

  case <- list(
    id = paste(paste0(criterion_name, "_", estimator_name), bank_name, examinees$labels[examinee], sep = "/"),
    bank = bank_name,
    examinee = examinee - 1L,
    criterion = criterion,
    method = estimator$method,
    rules = as_array(recorded$stop$rule),
    thresholds = as_array(recorded$stop$thr),
    items = as_array(as.integer(items) - 1L),
    thetas = as_array(thetas),
    ses = as_array(ses),
    se_slopes = as_array(slopes)
  )
  if (estimator$method %in% c("ML", "BM", "WL")) case$range <- estimator$range
  if (estimator$method %in% c("BM", "EAP")) case$prior <- list(distribution = estimator$priorDist, parameters = estimator$priorPar)
  if (estimator$method == "EAP") case$grid <- list(low = GRID[1], high = GRID[2], nodes = as.integer(GRID[3]))
  if (!is.null(estimator$constantPatt)) case$constant_pattern <- estimator$constantPatt
  case
}

whole_test_cases <- function() {
  cases <- list()
  add <- function(case) cases[[length(cases) + 1L]] <<- case
  examinees <- lapply(RECORDED_BANKS, recorded_examinees)
  for (bank_name in names(RECORDED_BANKS)) {
    for (criterion_name in names(WHOLE_TEST_CRITERIA)) {
      for (estimator_name in names(WHOLE_TEST_ESTIMATORS)) {
        for (examinee in seq_along(examinees[[bank_name]]$labels)) {
          add(whole_test_case(bank_name, examinees[[bank_name]], examinee, criterion_name, estimator_name))
        }
      }
    }
  }
  # catR's default start gives the same tests as no early item; startItems has MFI and bOpt rules,
  # not MLWI
  by_id <- setNames(cases, vapply(cases, function(case) case$id, ""))
  for (criterion_name in c("mfi", "bopt")) {
    for (estimator_name in c("ml", "eap")) {
      for (examinee in seq_along(examinees$selection$labels)) {
        early <- whole_test_case("selection", examinees$selection, examinee, criterion_name, estimator_name, start_items = TRUE)
        stopifnot(identical(early, by_id[[early$id]]))
      }
    }
  }
  list(
    banks = lapply(RECORDED_BANKS, function(recorded) recorded$bank),
    examinees = lapply(examinees, function(recorded) recorded$patterns == 1L),
    cases = cases
  )
}
