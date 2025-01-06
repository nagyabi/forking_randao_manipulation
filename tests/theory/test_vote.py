import itertools
import numpy as np
import pytest

from sympy import symbols, expand

from theory.method.utils.voting import dynamic_vote
from base.helpers import BOOST

slots_arg = list(range(1, 10))
nec_slots_args = list(range(1, 5))

arguments = list(itertools.product(slots_arg, nec_slots_args))


@pytest.mark.parametrize("slots, nec_slots", arguments)
def test_dynamic_vote(slots: int, nec_slots):
    if slots < nec_slots:
        return  # The attacker can't fork anyway

    alpha_min = np.float64((1 - BOOST) / (nec_slots + 2))
    alpha_max = np.float64((1 - BOOST) / (nec_slots + 1))
    alpha = (alpha_min + alpha_max) / 2
    _, voting = dynamic_vote(
        alpha=alpha,
        adv_cluster1=slots,
        honest_cluster=1,
        adv_cluster2=1,
        epoch_boundary=0,
    )
    plans = [(plan, vote.free_slots_before) for vote in voting for plan in vote.plans]
    assert all(plan[0].addit == 0 for plan in plans)

    sac = symbols("sac")
    expected = (1 + sac) ** slots - sac ** (slots - nec_slots + 1) * (1 + sac) ** (
        nec_slots - 1
    )
    expected = expand(expected)

    actual = 0
    for plan, free_slots_before in plans:
        adding = sac**plan.base_sacrifice
        adding *= (1 + sac) ** (plan.free_slots_after + free_slots_before)
        adding *= plan.possibilities_middle
        actual += adding

    actual = expand(actual)

    assert actual == expected, f"{slots=} {actual=}\n{plans=}"
