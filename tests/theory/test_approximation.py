from itertools import product
import numpy as np
import pytest
from theory.method.distribution import ApproximatedDistribution


dist1 = ApproximatedDistribution(distribution=np.array([1, 1.5, 2, 4, 4.4], dtype=np.float64))
dist2 = ApproximatedDistribution(distribution=np.array([3, 3.2, 8, 10, 11], dtype=np.float64))
dist3 = ApproximatedDistribution(distribution=np.array([0, 1.5, 4.44, 5, 13], dtype=np.float64))
dist4 = ApproximatedDistribution(distribution=np.array([-1, 4, 4.1, 4.2, 4.3], dtype=np.float64))

dists = [dist1, dist2, dist3, dist4]


@pytest.mark.parametrize("dist1, dist2", product(dists, dists))
def test_max_unknown(dist1: ApproximatedDistribution, dist2: ApproximatedDistribution):
    result = dist1.max_unknown(unknown=dist2)
    samples = np.random.choice(dist2.distribution, size=500_000)
    for elem, res_elem in zip(dist1.distribution, result.distribution):
        unk_max = np.average(np.maximum(elem, samples))
        assert np.isclose(res_elem, unk_max, rtol=0.07)

@pytest.mark.parametrize("dist1, dist2", product(dists, dists))
def test_max_exp(dist1: ApproximatedDistribution, dist2: ApproximatedDistribution):
    samples1 = np.random.choice(dist1.distribution, size=50_000)
    samples2 = np.random.choice(dist2.distribution, size=50_000)
    avg_experiment = np.average(np.maximum(samples1, samples2))
    avg_actual = dist1.max(dist2).expected_value()
    assert np.isclose(avg_actual, avg_experiment, rtol=0.07)