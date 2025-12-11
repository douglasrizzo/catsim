# Contributing

**catsim** is built in an object-oriented paradigm (or as object-oriented as Python allows) so it is rather simple to extend it. You can write new [initializers](https://douglasrizzo.com.br/catsim/initialization.html), [selectors](https://douglasrizzo.com.br/catsim/selection.html), [estimators](https://douglasrizzo.com.br/catsim/estimation.html) or [stopping criteria](https://douglasrizzo.com.br/catsim/stopping.html) by extending the base classes that are present in each of the corresponding modules. You can also write new [IRT-related functions](https://douglasrizzo.com.br/catsim/irt.html), as long as you have the right academic papers to prove they are relevant.

If you think the [simulator](https://douglasrizzo.com.br/catsim/simulation.html) could be doing something it is currently not doing, feel free to study it and make a contribution.

If you know a better way to present all the data collected during simulation, feel free to contribute with your own [plots](https://douglasrizzo.com.br/catsim/plot.html).

## Psychometrics

**catsim** still has a way to go before it can be considered a mature package. Here is a list of features that may be of interest:

- Bayesian ability estimators (maybe using [PyMC](https://www.pymc.io/));
- Other test evaluation methods;
- Comparisons between simulation results (for example [^Barr2010]);
- Other information functions, selection methods based on intervals or areas of information etc. (see [^Lind2010]).

## Unit testing

If you are interested in making a contribution to **catsim**, I'd encourage you to also contribute with unit tests in the package's testing module.

## How to contribute

**Contributing code:** create a fork on GitHub, make your changes on your own repo and then send a pull request to our **testing** branch so that we can check your contributions. Make sure your version passes on the unit tests.

**Contributing ideas:** file an [issue on GitHub](https://github.com/douglasrizzo/catsim/issues), label it as an **enhancement** and describe as thoroughly as possible what could be done.

**Blaming me:** file an [issue on GitHub](https://github.com/douglasrizzo/catsim/issues) describing as thoroughly as possible the problem, with error messages, descriptions of your tests and possibly suggestions for fixing it.

## References

[^Lind2010]: Linden, W. J. V. D., & Pashley, J. P. Item Selection and Ability Estimation in Adaptive Testing. _In_ Linden, W. J. V. D., & Glas, C. A. W. (2010). Elements of Adaptive Testing. New York, NY, USA: Springer New York.
[^Barr2010]: Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2010). A Method for the Comparison of Item Selection Rules in Computerized Adaptive Testing. Applied Psychological Measurement, 34(6), 438–452. <http://doi.org/10.1177/0146621610370152>
