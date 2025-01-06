from copy import deepcopy
import numpy as np
import pytest
from theory.selfish_mixing import SMValuedDistribution
from base.helpers import SLOTS_PER_EPOCH

EPSILON = np.float64(1e-7)
SIZE = 10_000_000


@pytest.mark.parametrize(
    "alpha",
    argvalues=[
        np.float64(0.01),
        np.float64(0.12),
        np.float64(0.15),
        np.float64(0.25),
        np.float64(0.44),
    ],
)
def test_basic_self_mixing_dist(alpha: np.float64):
    valued_distribution = SMValuedDistribution.make_binom(
        alpha=alpha,
        tail_slots_to_value={i: 0 for i in range(SLOTS_PER_EPOCH + 1)},
        sacrifice=0,
    )

    assert np.isclose(
        np.sum(valued_distribution.distribution), 1.0, rtol=EPSILON
    ), f"{np.sum(valued_distribution.distribution)=}"
    c_distribution = deepcopy(valued_distribution)
    expected_val = c_distribution.expected_value()
    c_less = c_distribution.increment_sacrifice()
    c_less_exp = c_less.expected_value()
    assert np.isclose(c_less_exp, expected_val - 1, rtol=EPSILON)
    smaller = SMValuedDistribution.make_binom(
        alpha=alpha,
        tail_slots_to_value={i: 0 for i in range(SLOTS_PER_EPOCH + 1)},
        sacrifice=1,
    )
    assert smaller == c_less
    assert c_distribution == valued_distribution


def num_of_bits(num: int) -> int:
    res = 0
    while num > 0:
        res += num % 2
        num //= 2
    return res


@pytest.mark.parametrize(
    "alpha",
    argvalues=[
        np.float64(0.01),
        np.float64(0.12),
        np.float64(0.15),
        np.float64(0.25),
        np.float64(0.44),
    ],
)
def test_max_with_simulation(alpha: np.float64):
    result: SMValuedDistribution = SMValuedDistribution.make_binom(
        alpha=alpha,
        tail_slots_to_value={i: 0 for i in range(SLOTS_PER_EPOCH + 1)},
        sacrifice=0,
    )

    for t in range(4):
        assert np.isclose(np.sum(result.distribution), 1, rtol=EPSILON)
        RO_size = 2**t
        rand_sample = np.random.binomial(
            n=SLOTS_PER_EPOCH, p=alpha, size=(RO_size, SIZE)
        )
        sacrifice = np.array([[num_of_bits(i)] * SIZE for i in range(RO_size)])
        ROs = rand_sample - sacrifice
        best_ROs = np.max(ROs, axis=0)
        simulated_avg = np.average(best_ROs)
        dist_exp_val = result.expected_value()
        assert np.isclose(
            dist_exp_val, simulated_avg, rtol=np.float64(5e-3)
        ), f"{alpha=} {dist_exp_val=} {simulated_avg=} {t=}"

        result_less = result.increment_sacrifice()
        result: SMValuedDistribution = result.max(result_less)
