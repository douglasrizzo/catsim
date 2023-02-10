import random

import numpy as np
import pytest
from sklearn.cluster import KMeans

from catsim import cat, irt, plot
from catsim.cat import generate_item_bank
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer, RandomInitializer
from catsim.selection import (
    AStratBBlockSelector,
    AStratSelector,
    ClusterSelector,
    LinearSelector,
    MaxInfoBBlockSelector,
    MaxInfoSelector,
    MaxInfoStratSelector,
    RandomesqueSelector,
    RandomSelector,
    The54321Selector,
    UrrySelector,
)
from catsim.simulation import Estimator, Initializer, Selector, Simulator, Stopper
from catsim.stopping import MaxItemStopper, MinErrorStopper


def one_simulation(items, examinees, initializer, selector, estimator, stopper):
    s = Simulator(items, examinees)
    s.simulate(initializer, selector, estimator, stopper, verbose=True)
    cat.rmse(s.examinees, s.latest_estimations)


@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("bank_size", [500])
@pytest.mark.parametrize("initializer", [RandomInitializer("uniform", (-5, 5))])
@pytest.mark.parametrize("estimator", [NumericalSearchEstimator()])
@pytest.mark.parametrize("stopper", [MaxItemStopper(30), MinErrorStopper(0.4)])
def test_cism(
    examinees: int,
    bank_size: int,
    initializer: Initializer,
    estimator: Estimator,
    stopper: Stopper,
):
    items = generate_item_bank(bank_size)
    clusters = list(KMeans(n_clusters=8, n_init="auto").fit_predict(items))
    ClusterSelector.weighted_cluster_infos(0, items, clusters)
    ClusterSelector.avg_cluster_params(items, clusters)
    selector = ClusterSelector(clusters=clusters, r_max=0.2)
    one_simulation(items, examinees, initializer, selector, estimator, stopper)


@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("test_size", [30])
@pytest.mark.parametrize("bank_size", [500])
@pytest.mark.parametrize("logistic_model", ["4PL"])
@pytest.mark.parametrize(
    "initializer",
    [
        RandomInitializer("uniform", (-5, 5)),
        FixedPointInitializer(0), ],
)
@pytest.mark.parametrize(
    "estimator", [NumericalSearchEstimator(method=m) for m in NumericalSearchEstimator.methods])
def test_finite_selectors(
    examinees: int,
    test_size: int,
    bank_size: int,
    logistic_model: str,
    initializer: Initializer,
    estimator: Estimator,
):
    finite_selectors = [
        LinearSelector(list(np.random.choice(bank_size, size=test_size, replace=False))),
        AStratSelector(test_size),
        AStratBBlockSelector(test_size),
        MaxInfoStratSelector(test_size),
        MaxInfoBBlockSelector(test_size),
        The54321Selector(test_size),
        RandomesqueSelector(test_size / 6), ]
    stopper = MaxItemStopper(test_size)

    for selector in finite_selectors:
        items = generate_item_bank(bank_size, itemtype=logistic_model)
        responses = cat.random_response_vector(random.randint(1, test_size - 1))
        administered_items = np.random.choice(bank_size, len(responses), replace=False)
        est_theta = initializer.initialize()
        selector.select(
            items=items,
            administered_items=administered_items,
            est_theta=est_theta,
        )
        estimator.estimate(
            items=items,
            administered_items=administered_items,
            response_vector=responses,
            est_theta=est_theta,
        )
        stopper.stop(
            administered_items=items[administered_items],
            theta=est_theta,
        )

        one_simulation(items, examinees, initializer, selector, estimator, stopper)


@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("test_size", [30])
@pytest.mark.parametrize("bank_size", [500])
@pytest.mark.parametrize("logistic_model", ["4PL"])
@pytest.mark.parametrize(
    "initializer",
    [
        RandomInitializer("uniform", (-5, 5)),
        FixedPointInitializer(0), ],
)
@pytest.mark.parametrize(
    "selector",
    [
        MaxInfoSelector(),
        RandomSelector(),
        UrrySelector(), ],
)
@pytest.mark.parametrize(
    "estimator", [NumericalSearchEstimator(method=m) for m in NumericalSearchEstimator.methods])
@pytest.mark.parametrize(
    "stopper",
    [
        MaxItemStopper(30),
        MinErrorStopper(0.4), ],
)
def test_infinite_selectors(
    examinees: int,
    test_size: int,
    bank_size: int,
    logistic_model: str,
    initializer: Initializer,
    selector: Selector,
    estimator: Estimator,
    stopper: Stopper,
):
    items = generate_item_bank(bank_size, itemtype=logistic_model)
    responses = cat.random_response_vector(random.randint(1, test_size - 1))
    administered_items = np.random.choice(bank_size, len(responses), replace=False)
    est_theta = initializer.initialize()
    selector.select(
        items=items,
        administered_items=administered_items,
        est_theta=est_theta,
    )
    estimator.estimate(
        items=items,
        administered_items=administered_items,
        response_vector=responses,
        est_theta=est_theta,
    )
    stopper.stop(
        administered_items=items[administered_items],
        theta=est_theta,
    )

    one_simulation(items, examinees, initializer, selector, estimator, stopper)


def test_item_bank_generation():
    for items in [
            generate_item_bank(5, "1PL"),
            generate_item_bank(5, "2PL"),
            generate_item_bank(5, "3PL"),
            generate_item_bank(5, "3PL", corr=0),
            generate_item_bank(5, "4PL"), ]:
        irt.validate_item_bank(items, raise_err=True)
    items = np.zeros(100)
    irt.validate_item_bank(items)
    items = irt.normalize_item_bank(items)
    irt.validate_item_bank(items, raise_err=True)


def test_plots():
    from matplotlib.pyplot import close

    initializer = RandomInitializer()
    selector = MaxInfoSelector()
    estimator = NumericalSearchEstimator()
    stopper = MaxItemStopper(20)
    s = Simulator(generate_item_bank(100), 10)
    s.simulate(initializer, selector, estimator, stopper, verbose=True)

    for item in s.items[0:10]:
        plot.item_curve(item[0], item[1], item[2], item[3], "Test plot", "icc", False, None, False)
        plot.item_curve(item[0], item[1], item[2], item[3], "Test plot", "iic", True, None, False)
        plot.item_curve(item[0], item[1], item[2], item[3], "Test plot", "both", True, None, False)
        close("all")

    plot.gen3d_dataset_scatter(items=s.items, show=False)
    plot.param_dist(items=s.items, show=False)
    plot.test_progress(
        title="Test progress",
        simulator=s,
        index=0,
        info=True,
        see=True,
        reliability=True,
        show=False,
    )
    plot.item_exposure(simulator=s, show=False)
    plot.item_exposure(simulator=s, show=False, par="a")
    plot.item_exposure(simulator=s, show=False, par="b")
    plot.item_exposure(simulator=s, show=False, par="c")
    plot.item_exposure(simulator=s, show=False, par="d")
    plot.item_exposure(simulator=s, show=False, hist=True)

    # close all plots after testing
    close("all")
