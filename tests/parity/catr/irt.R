# Response probabilities (catR's Pi) and item information (Ii) of 1PL to 4PL items.
#
# catR's scaling constant D stays at its default 1, the logistic catsim uses. The abilities keep
# every logit a (theta - b) at most 10: above it catR computes 1 - P by subtracting a P close to 1
# and loses the relative precision of the comparison.

IRT_THETAS <- c(-4, -2.3, -0.75, 0, 0.4, 1.6, 3.2)

IRT_BANKS <- list(
  `1pl` = cbind(1, c(-2.1, -0.65, 0, 0.8, 1.9), 0, 1),
  `2pl` = cbind(c(0.6, 1.1, 1.8, 2.4, 0.9), c(-1.4, 0.3, -0.2, 1.1, 2.2), 0, 1),
  `3pl` = cbind(c(0.6, 1.1, 1.8, 2.4, 0.9), c(-1.4, 0.3, -0.2, 1.1, 2.2), c(0.15, 0.2, 0.25, 0.1, 0.3), 1),
  `4pl` = cbind(
    c(0.6, 1.1, 1.8, 2.4, 0.9), c(-1.4, 0.3, -0.2, 1.1, 2.2), c(0.15, 0.2, 0.25, 0.1, 0.3),
    c(0.95, 0.9, 0.85, 0.97, 0.92)
  )
)

irt_cases <- function() {
  cases <- lapply(names(IRT_BANKS), function(model) {
    items <- IRT_BANKS[[model]]
    stopifnot(max(sweep(outer(IRT_THETAS, items[, 2], "-"), 2, items[, 1], "*")) <= 10)
    list(
      id = model,
      items = items,
      thetas = IRT_THETAS,
      probability = t(sapply(IRT_THETAS, function(theta) Pi(theta, items)$Pi)),
      information = t(sapply(IRT_THETAS, function(theta) Ii(theta, items)$Ii))
    )
  })
  list(cases = cases)
}
