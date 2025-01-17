from copy import deepcopy
from itertools import product
from math import comb
from types import MappingProxyType
import numpy as np
import pytest

from theory.method.utils.attack_strings import ExtendedAttackString
from theory.method.detailed_distribution import (
    DetailedDistribution,
    KnownCondition,
    DetailedElement,
    Outcome,
    Reason,
)
from theory.method.distribution import (
    DetailedDistributionMaker,
    DistributionMaxer,
    Element,
    RichDistributionMaker,
    RichElement,
    RichValuedDistribution,
    T_Distribution,
    ValuedDistribution,
    ValuedDistributionMaker,
)
from theory.method.preeval import (
    calc_attack_string_mapping,
    calc_eas_mapping,
    calc_postfix_mapping,
)
from base.helpers import SLOTS_PER_EPOCH
from scipy.stats import binom

EPSILON = np.float64(1e-10)
SIZE = 50_000_000

dist1 = ValuedDistribution(
    distribution=[
        Element(
            value=np.float64(-4.0),
            probability=np.float64(0.4),
        ),
        Element(
            value=np.float64(1.0),
            probability=np.float64(0.6),
        ),
    ]
)

dist1_det = DetailedDistribution(
    slot=-1,
    distribution=(
        DetailedElement(
            value=np.float64(-4.0),
            probability=np.float64(0.4),
            reasons=(
                Reason(
                    condition=KnownCondition(adv_slots=4, epoch_str="a#ah"),
                    id=0,
                ),
            ),
        ),
        DetailedElement(
            value=np.float64(1.0),
            probability=np.float64(0.6),
            reasons=(
                Reason(
                    condition=KnownCondition(adv_slots=5, epoch_str="a#aaah"),
                    id=0,
                ),
            ),
        ),
    ),
    id_to_outcome=MappingProxyType(
        {
            0: Outcome(
                config=0,
                full_config=0,
                end_slot=1,
                sacrifice=1,
                known=True,
                regret=False,
                det_slot_statuses=(),
            )
        }
    ),
)

dist2 = ValuedDistribution(
    distribution=[
        Element(
            value=np.float64(-2.0),
            probability=np.float64(0.1),
        ),
        Element(
            value=np.float64(1.0),
            probability=np.float64(0.4),
        ),
        Element(
            value=np.float64(9.0),
            probability=np.float64(0.5),
        ),
    ]
)

dist2_det = DetailedDistribution(
    slot=-1,
    distribution=(
        DetailedElement(
            value=np.float64(-2.0),
            probability=np.float64(0.1),
            reasons=(
                Reason(
                    condition=KnownCondition(adv_slots=2, epoch_str="#"),
                    id=1,
                ),
            ),
        ),
        DetailedElement(
            value=np.float64(1.0),
            probability=np.float64(0.4),
            reasons=(
                Reason(
                    condition=KnownCondition(adv_slots=3, epoch_str="a#"),
                    id=1,
                ),
            ),
        ),
        DetailedElement(
            value=np.float64(9.0),
            probability=np.float64(0.5),
            reasons=(
                Reason(
                    condition=KnownCondition(adv_slots=7, epoch_str="a#aha"),
                    id=1,
                ),
            ),
        ),
    ),
    id_to_outcome=MappingProxyType(
        {
            1: Outcome(
                config=1,
                full_config=4,
                end_slot=1,
                sacrifice=2,
                known=True,
                regret=False,
                det_slot_statuses=(),
            )
        }
    ),
)

dist11 = ValuedDistribution(
    distribution=[
        Element(
            value=np.float64(-4.0),
            probability=np.float64(0.16),
        ),
        Element(
            value=np.float64(1.0),
            probability=np.float64(0.84),
        ),
    ]
)

dist22 = ValuedDistribution(
    distribution=[
        Element(
            value=np.float64(-2.0),
            probability=np.float64(0.01),
        ),
        Element(
            value=np.float64(1.0),
            probability=np.float64(0.24),
        ),
        Element(
            value=np.float64(9.0),
            probability=np.float64(0.75),
        ),
    ]
)


dist12 = ValuedDistribution(
    distribution=[
        Element(
            value=np.float64(-2.0),
            probability=np.float64(0.04),
        ),
        Element(
            value=np.float64(1.0),
            probability=np.float64(0.46),
        ),
        Element(
            value=np.float64(9.0),
            probability=np.float64(0.5),
        ),
    ]
)

arguments1 = [
    (dist1, np.float64(-1.0)),
    (dist2, np.float64(4.7)),
]

arguments2 = [
    (dist1, dist1, dist11),
    (dist1, dist2, dist12),
    (dist2, dist1, dist12),
    (dist2, dist2, dist22),
]


@pytest.mark.parametrize("dist1, val", arguments1)
def test_valued_distribution_basics(dist1: ValuedDistribution, val: np.float64):
    assert np.isclose(dist1.expected_value(), val), f"{dist1.expected_value=} {val=}"
    shifted = dist1.increase_sacrifice(sacrifice=2)
    assert np.isclose(
        shifted.expected_value(), val - 2
    ), f"{shifted.expected_value()=} {val-2=}"


@pytest.mark.parametrize("dist1, dist2, max_dist", arguments2)
def test_valued_distribution_max(
    dist1: ValuedDistribution, dist2: ValuedDistribution, max_dist: ValuedDistribution
):
    max_act = dist1.max(dist2)
    assert dist2.max(dist1) == max_act

    assert len(max_dist.distribution) == len(max_act.distribution)
    for a, b in zip(max_act.distribution, max_dist.distribution):
        assert np.isclose(a.probability, b.probability, rtol=EPSILON), f"{a=} {b=}"
        assert a.value == b.value


arguments_shadow = [
    (dist1, dist1_det, dist2, dist2_det),
    (dist2, dist2_det, dist1, dist1_det),
]


@pytest.mark.parametrize("dist1, shadow1, dist2, shadow2", arguments_shadow)
def test_detailed_dist_shadowing(
    dist1: ValuedDistribution,
    shadow1: DetailedDistribution,
    dist2: ValuedDistribution,
    shadow2: DetailedDistribution,
) -> None:
    """
    This test aims to test whether the basic features of DetailedDistribution works like ValuedDistribution.
    Instead of testing the additional feature, we make sure the core functionality works well.
    Only valid test if ValuedDistribution works.

    Args:
        dist1 (ValuedDistribution): first distribution
        shadow1 (DetailedDistribution): shadow of the 1st dist, with the same probabilities and values
        dist2 (ValuedDistribution): second distribution
        shadow2 (DetailedDistribution): shadow of the 2nd dist, with the same probabilities and values
    """

    def similar(dist: ValuedDistribution, shadow: DetailedDistribution) -> None:
        assert len(dist.distribution) == len(shadow.distribution)
        for elem, d_elem in zip(dist.distribution, shadow.distribution):
            assert np.isclose(elem.value, d_elem.value, rtol=EPSILON)
            assert np.isclose(elem.probability, d_elem.probability, rtol=EPSILON)

    similar(dist=dist1, shadow=shadow1)
    similar(dist=dist2, shadow=shadow2)

    sac = dist1.increase_sacrifice(1)
    sac_sh = shadow1.insert_noncanonical(True)

    sac_nomatch = shadow1.insert_noncanonical(False)

    similar(dist=sac, shadow=sac_sh)

    with pytest.raises(AssertionError):
        similar(dist=sac, shadow=sac_nomatch)

    maxi = dist1.max(dist2)
    maxi_sh = shadow1.max(shadow2)

    similar(dist=maxi, shadow=maxi_sh)

    unk = dist1.max_unknown(dist2)
    unk_sh = shadow1.max_unknown(shadow2)

    similar(dist=unk, shadow=unk_sh)


arguments_um = [4, 5, 10]


@pytest.mark.parametrize("n", arguments_um)
def test_valued_distribution_unknown_max_with_simulation(n: int):

    distribution_x = ValuedDistribution(
        distribution=[
            Element(
                value=np.float64(i),
                probability=np.float64(1 / n),
            )
            for i in range(n)
        ]
    )
    distribution_y = deepcopy(distribution_x)
    distribution_z = distribution_x.max_unknown(unknown=distribution_y)

    x = np.random.randint(0, n, SIZE)  # known distribution
    y = np.random.randint(0, n, SIZE)  # unknown distribution
    z = np.maximum(x, y)

    stats = np.zeros(n, dtype=np.float64)

    for num in range(n):
        positive = np.isclose(x, num)
        stats[num] = np.sum(positive * z) / np.sum(positive)

    # TEST: Is the experienced distribution similar enough to the calculated one
    assert np.all(
        np.isclose(
            stats, [elem.value for elem in distribution_z.distribution], rtol=1e-2
        )
    ), f"{stats=} {distribution_z.distribution=}"


arguments_maker = [
    (np.float64(0.1), 1, 1, "a"),
    (np.float64(0.2), 3, 2, "aa"),
    (np.float64(0.3), 5, 3, ""),
]


@pytest.mark.parametrize(
    "alpha,size_prefix,size_postfix,epoch_postfix", arguments_maker
)
def test_valued_distribution_maker(
    alpha: np.float64, size_prefix: int, size_postfix: int, epoch_postfix: str
):
    as_to_as = calc_attack_string_mapping(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
    )
    eas_mapping = calc_eas_mapping(as_to_as[1], calc_postfix_mapping(as_to_as[1]))
    maker = ValuedDistributionMaker(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        eas_mapping=eas_mapping,
    )
    rich_maker = RichDistributionMaker(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        eas_mapping=eas_mapping,
    )
    value_function = {
        str(eas): np.float64(0.0)
        for eas in ExtendedAttackString.possibilities(
            size_prefix=size_prefix, size_postfix=size_postfix
        )
    }

    distribution = maker.make_distribution(
        value_function=value_function, postfix_next_epoch=epoch_postfix
    )
    rich_distribution = rich_maker.make_distribution(
        value_function=value_function, postfix_next_epoch=epoch_postfix
    )
    detailed_maker = DetailedDistributionMaker(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        eas_mapping=eas_mapping,
    )

    detailed = detailed_maker.make_detailed(
        value_function=value_function, postfix_next_epoch=epoch_postfix, nullcase=".a"
    )
    assert len(detailed.distribution) == len(distribution.distribution)
    all_combos: set[tuple[int, str]] = set()
    for elem, d_elem in zip(distribution.distribution, detailed.distribution):
        assert np.isclose(elem.value, d_elem.value, rtol=EPSILON)
        assert np.isclose(elem.probability, d_elem.probability, rtol=EPSILON)
        for reason in d_elem.reasons:

            assert isinstance(reason.condition, KnownCondition)

            e = (reason.condition.adv_slots, reason.condition.epoch_str)
            assert e not in all_combos
            all_combos.add(e)

    assert np.isclose(
        sum(elem.probability for elem in distribution.distribution),
        np.float64(1.0),
        rtol=EPSILON,
    )
    assert np.isclose(
        sum(elem.probability for elem in rich_distribution.distribution),
        np.float64(1.0),
        rtol=EPSILON,
    ), f"{sum(elem.probability for elem in rich_distribution.distribution)=} {rich_distribution.distribution=} {distribution.distribution=}"
    assert np.all(
        np.isclose(
            [elem.probability for elem in distribution.distribution],
            binom.pmf(np.arange(SLOTS_PER_EPOCH + 1), SLOTS_PER_EPOCH, alpha),
            rtol=EPSILON,
        )
    )
    assert np.all(
        np.isclose(
            [elem.probability for elem in rich_distribution.distribution],
            binom.pmf(np.arange(SLOTS_PER_EPOCH + 1), SLOTS_PER_EPOCH, alpha),
            rtol=EPSILON,
        )
    )
    assert np.all(
        [
            np.isclose(
                elem.probability,
                np.sum(elem.distribution_of_epoch_strings),
                rtol=EPSILON,
            )
            for elem in rich_distribution.distribution
        ]
    )
    assert np.isclose(
        np.sum(
            [elem.expected_value_of_vars[0] for elem in rich_distribution.distribution]
        ),
        SLOTS_PER_EPOCH * alpha,
    )

    rich_distribution = rich_distribution.max(rich_distribution.increase_sacrifice(2))

    assert [
        np.isclose(
            elem.probability, np.sum(elem.distribution_of_epoch_strings), rtol=EPSILON
        )
        for elem in rich_distribution.distribution
    ]
    rich_exp = rich_distribution.expected_value_in_distribution()
    assert np.isclose(
        rich_exp.expected_value(), rich_distribution.expected_value(), rtol=EPSILON
    )
    assert np.isclose(
        rich_exp.distribution[0].expected_value_of_vars[0],
        rich_distribution.expected_value(),
        rtol=EPSILON,
    )
    assert np.isclose(
        sum(
            np.sum(elem.distribution_of_epoch_strings)
            for elem in rich_distribution.distribution
        ),
        1.0,
        rtol=EPSILON,
    )

    rich_distribution = rich_distribution.max_unknown(rich_distribution)

    assert [
        np.isclose(
            elem.probability, np.sum(elem.distribution_of_epoch_strings), rtol=EPSILON
        )
        for elem in rich_distribution.distribution
    ]
    rich_exp = rich_distribution.expected_value_in_distribution()
    assert np.isclose(
        rich_exp.expected_value(), rich_distribution.expected_value(), rtol=EPSILON
    )
    assert np.isclose(
        rich_exp.distribution[0].expected_value_of_vars[0],
        rich_distribution.expected_value(),
        rtol=EPSILON,
    )
    assert np.isclose(
        sum(
            np.sum(elem.distribution_of_epoch_strings)
            for elem in rich_distribution.distribution
        ),
        1.0,
        rtol=EPSILON,
    )


a0 = ValuedDistribution(
    distribution=[
        Element(value=np.float64(2.0), probability=np.float64(0.3)),
        Element(value=np.float64(7.0), probability=np.float64(0.4)),
        Element(value=np.float64(8.0), probability=np.float64(0.3)),
    ]
)

a1 = ValuedDistribution(
    distribution=[
        Element(value=np.float64(2.0), probability=np.float64(0.1)),
        Element(value=np.float64(2.1), probability=np.float64(0.6)),
        Element(value=np.float64(7.0), probability=np.float64(0.2)),
        Element(value=np.float64(8.0), probability=np.float64(0.1)),
    ]
)

b1 = ValuedDistribution(
    distribution=[
        Element(value=np.float64(0.0), probability=np.float64(0.2)),
        Element(value=np.float64(1.1), probability=np.float64(0.2)),
        Element(value=np.float64(4.3), probability=np.float64(0.2)),
        Element(value=np.float64(9.0), probability=np.float64(0.3)),
        Element(value=np.float64(9.9), probability=np.float64(0.1)),
    ]
)

a2 = RichValuedDistribution(
    distribution=[
        RichElement(
            value=np.float64(1.5),
            probability=np.float64(0.3),
            expected_value_of_vars=np.array(
                [0.03, 1.99, 69.2, 0.3, 0.2, 0.1, 0.004, 0.00044, 0.000076],
                dtype=np.float64,
            ),
            distribution_of_epoch_strings=np.array([0.2, 0.1], dtype=np.float64),
        ),
        RichElement(
            value=np.float64(2.66),
            probability=np.float64(0.5),
            expected_value_of_vars=np.array(
                [0.04, -3, -10, 0.0, 0.1, 0.3, 0.009, 0.00014, 0.000079],
                dtype=np.float64,
            ),
            distribution_of_epoch_strings=np.array([0.1, 0.4], dtype=np.float64),
        ),
        RichElement(
            value=np.float64(3.9),
            probability=np.float64(0.2),
            expected_value_of_vars=np.array(
                [0.03, 134.45, 18.5, 0.4, 0.2, 0.1, 0.001, 0.0, 0.000111],
                dtype=np.float64,
            ),
            distribution_of_epoch_strings=np.array([0.05, 0.15], dtype=np.float64),
        ),
    ]
)

b2 = RichValuedDistribution(
    distribution=[
        RichElement(
            value=np.float64(0.5),
            probability=np.float64(0.2),
            expected_value_of_vars=np.array(
                [0.09, 1.1, -9.2, 1.0, 0.0, 0.0, 0.002, 0.01, 0.0003], dtype=np.float64
            ),
            distribution_of_epoch_strings=np.array([0.2, 0.0], dtype=np.float64),
        ),
        RichElement(
            value=np.float64(2.71828),
            probability=np.float64(0.1),
            expected_value_of_vars=np.array(
                [4.04, -6, -4.3, 0.12, 0.18, 0.7, 1.12, 0.66, 0.0], dtype=np.float64
            ),
            distribution_of_epoch_strings=np.array([0.05, 0.05], dtype=np.float64),
        ),
        RichElement(
            value=np.float64(3.4),
            probability=np.float64(0.7),
            expected_value_of_vars=np.array(
                [8.0, 0, 50, 0.1, 0.1, 0.2, 2.0, 1.0, 5.0], dtype=np.float64
            ),
            distribution_of_epoch_strings=np.array([0.6, 0.1], dtype=np.float64),
        ),
    ]
)

arguments_imm = [
    (a1, a1),
    (a1, b1),
    (b1, a1),
    (b1, b1),
    (a2, a2),
    (a2, b2),
    (b2, a2),
    (b2, b2),
]


@pytest.mark.parametrize("a, b", arguments_imm)
def test_distribution_immutability(a: T_Distribution, b: T_Distribution):
    """
    Technically the distribution implementation is not immutable and instances can share the same state.
    We would like to test out whether all the used methods use/effect 'a'
    """
    a_copy = deepcopy(a)
    assert a == a_copy
    a.max(a)
    assert a == a_copy
    a.max(b)
    assert a == a_copy
    b.max(b)
    assert a == a_copy
    b.max(a)
    assert a == a_copy
    a.expected_value()
    assert a == a_copy
    a_exp = a.expected_value_in_distribution()
    assert a == a_copy
    a_exp.max(a)
    assert a == a_copy
    a_exp.max(b)
    assert a == a_copy
    a.max(a_exp)
    assert a == a_copy
    a_unk = a.max_unknown(a)
    assert a == a_copy
    a_unk.max(a)
    assert a == a_copy
    a_unk.max_unknown(a)
    assert a == a_copy
    a_unk.expected_value_in_distribution()
    assert a == a_copy
    a.max_unknown(b)
    assert a == a_copy
    b.max_unknown(a)
    assert a == a_copy
    a_unk.max_unknown(b)
    assert a == a_copy
    a_inc = a.increase_sacrifice(1)
    assert a == a_copy
    a_inc.max(a)
    assert a == a_copy
    a_inc.max(b)
    assert a == a_copy
    a_inc.max_unknown(a.increase_sacrifice(4))
    assert a == a_copy
    a.max_unknown(a)
    assert a == a_copy
    b.max_unknown(a_inc)
    assert a == a_copy
    a.max(a_inc)
    assert a == a_copy


arguments_comm = product([a0, a1, a2, b1, b2], list(range(1, 15)))


@pytest.mark.parametrize("distribution, slots", arguments_comm)
def test_commutativity(slots: int, distribution: ValuedDistribution):
    """
    We test the commutativity of the distribution
    Args:
        slots (int): number of slots for selfish mixing
    """

    fast_dist = distribution
    for _ in range(slots):
        fast_dist = fast_dist.max(fast_dist.increase_sacrifice(1))
    fast_exp_val = fast_dist.expected_value()

    pascal_triangle = [comb(slots, i) for i in range(slots + 1)]
    slow_dist = distribution
    for sac, group in enumerate(pascal_triangle[1:], start=1):
        maxer = DistributionMaxer(distribution.increase_sacrifice(sac))
        maxed = maxer.max(group)
        slow_dist = slow_dist.max(maxed)

    slow_exp_val = slow_dist.expected_value()

    assert np.isclose(
        fast_exp_val, slow_exp_val, rtol=1e-8
    ), f"{fast_exp_val=} {slow_exp_val=}"
    assert len(fast_dist.distribution) == len(slow_dist.distribution)
    for fast_el, slow_el in zip(fast_dist.distribution, slow_dist.distribution):
        assert np.isclose(fast_el.probability, slow_el.probability, rtol=1e-8)
        assert np.isclose(fast_el.value, slow_el.value, rtol=1e-8)


arguments_maxer = [
    (a1, 4),
    (a1, 7),
    (a1, 6),
    (a1, 3),
]


@pytest.mark.parametrize("distr, times", arguments_maxer)
def test_maxer(distr: T_Distribution, times: int):
    maxer = DistributionMaxer(distr)
    fast_m = maxer.max(number_of_dist=times)
    fast_exp_val = fast_m.expected_value()

    slow_dist = distr
    for _ in range(times - 1):
        slow_dist = slow_dist.max(distr)
    slow_exp_val = slow_dist.expected_value()

    assert np.isclose(
        fast_exp_val, slow_exp_val, rtol=1e-7
    ), f"{fast_exp_val=} {slow_exp_val=}"
