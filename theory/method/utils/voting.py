from dataclasses import dataclass
import numpy as np

from base.helpers import BOOST


@dataclass
class Plan:
    free_slots_after: int
    base_sacrifice: int
    possibilities_middle: int

    addit: int


@dataclass
class VoteResult:
    regret_sac: int
    free_slots_before: int
    plans: list[Plan]


def dynamic_vote(
    alpha: np.float64,
    adv_cluster1: int,
    honest_cluster: int,
    adv_cluster2: int,
    epoch_boundary: int,
) -> tuple[int, list[VoteResult]]:
    """
    Situation: "a" * adv_cluster1 + "h" * honest_cluster + "a" * adv_cluster2.

    Args:
        alpha (np.float64): proportional stake of the attacker.
        adv_cluster1 (int): number adv blocks in the first cluster
        honest_cluster (int): number of honest blocks between the 2 adv cluster
        adv_cluster2 (int): number of adv blocks in the second cluster
        epoch_boundary (int): non-positive number regarding where is the epoch boundary

    Returns:
        tuple[int, list[VoteResult]]: The number of max addit + 1, The regret distribution is coming from the same distribution with differing sacrifices
    """
    assert adv_cluster1 > 0 and honest_cluster > 0 and adv_cluster2 > 0
    assert (
        -epoch_boundary - 1 < honest_cluster + adv_cluster2
    ), f"{epoch_boundary=} {adv_cluster1=} {honest_cluster=} {adv_cluster2=}"
    assert (
        -epoch_boundary != honest_cluster + adv_cluster2
    ), f"{epoch_boundary=} {adv_cluster1=} {honest_cluster=} {adv_cluster2=}"
    proposer_boost = np.float64(BOOST)

    additional = 0
    additional_to_necessary_slots: dict[int, int] = {}

    first = True
    while (
        (honest_cluster + additional - proposer_boost)
        / (adv_cluster1 + 2 * (honest_cluster + additional))
        < alpha
        and additional < adv_cluster2
        and (first or -epoch_boundary <= adv_cluster2 - additional)
    ):
        first = False
        more_slots_than_this = (
            (honest_cluster + additional) * (1 - 2 * alpha) - proposer_boost
        ) / alpha
        additional_to_necessary_slots[additional] = int(more_slots_than_this) + 1
        assert additional_to_necessary_slots[additional] > 0
        assert additional_to_necessary_slots[additional] <= adv_cluster1

        additional += 1

    if len(additional_to_necessary_slots) == 0:
        return additional, []

    result: list[VoteResult] = []
    for proposed_in_regret in range(2, adv_cluster1 + 1):
        addits = [
            addit
            for addit, nec_slots in additional_to_necessary_slots.items()
            if proposed_in_regret >= nec_slots
        ]
        new_elem = VoteResult(
            regret_sac=proposed_in_regret - 1,
            free_slots_before=adv_cluster1 - proposed_in_regret,
            plans=[],
        )
        for addit in addits:
            diff = proposed_in_regret - additional_to_necessary_slots[addit]
            if diff > 0:
                new_elem.plans.append(
                    Plan(
                        free_slots_after=additional_to_necessary_slots[addit] - 1,
                        base_sacrifice=diff - 1,
                        possibilities_middle=diff,
                        addit=addit,
                    )
                )
        if len(new_elem.plans) > 0:
            result.append(new_elem)
    addits = list(additional_to_necessary_slots)
    new_elem = VoteResult(
        regret_sac=adv_cluster1,
        free_slots_before=0,
        plans=[],
    )
    for addit in addits:
        new_elem.plans.append(
            Plan(
                free_slots_after=additional_to_necessary_slots[addit] - 1,
                base_sacrifice=adv_cluster1 - additional_to_necessary_slots[addit],
                possibilities_middle=adv_cluster1
                - additional_to_necessary_slots[addit]
                + 1,
                addit=addit,
            )
        )
    if len(new_elem.plans) > 0:
        result.append(new_elem)
    return additional, result
