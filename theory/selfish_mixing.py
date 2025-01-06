"""
This file aims to reproduce the result's of the paper: 'Optimal RANDAO Manipulation in Ethereum'.
We were able to approximate the results with an error margin of 1e-5.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from typing import Optional
import numpy as np

from scipy.stats import binom
from tqdm import tqdm

from base.helpers import CACHE_FOLDER, SLOTS_PER_EPOCH


@dataclass(frozen=True)
class SMEpochString:
    tail_slots: int
    RO: int
    value: np.float64

    def __eq__(self, value: object) -> bool:
        if isinstance(value, SMEpochString):
            return self.tail_slots == value.tail_slots and self.RO == value.RO
        return False

    def __hash__(self) -> int:
        return hash((self.tail_slots, self.RO))


@dataclass
class SMValuedDistribution:
    distribution: np.ndarray
    state_to_vf: list[SMEpochString]
    vf_to_index: dict[SMEpochString, int]
    heur_max: int = SLOTS_PER_EPOCH  # For measuring theory's heuristics error

    def __eq__(self, value: object) -> bool:
        if isinstance(value, SMValuedDistribution):
            return (
                np.all(self.distribution == value.distribution)
                and self.state_to_vf == value.state_to_vf
                and self.vf_to_index == value.vf_to_index
            )
        return False

    def increment_sacrifice(self) -> "SMValuedDistribution":
        new_distribution = np.zeros_like(self.distribution)
        for index, vf in enumerate(self.state_to_vf):
            smaller = SMEpochString(
                tail_slots=vf.tail_slots,
                RO=vf.RO - 1,
                value=vf.value - 1,
            )
            if smaller in self.vf_to_index:
                smaller_index = self.vf_to_index[smaller]
                assert new_distribution[smaller_index] == 0.0
                new_distribution[smaller_index] = self.distribution[index]
            else:
                assert self.distribution[index] == 0.0, f"{self.distribution[index]=}"

        return SMValuedDistribution(
            distribution=new_distribution,
            state_to_vf=self.state_to_vf,
            vf_to_index=self.vf_to_index,
            heur_max=self.heur_max,
        )

    def expected_value(self) -> np.float64:
        return np.dot(self.distribution, [vf.value for vf in self.state_to_vf])

    def dist_of_RO(self) -> np.ndarray:
        result = np.zeros(SLOTS_PER_EPOCH + 1, dtype=np.float64)
        assert len(self.distribution) == len(self.state_to_vf)
        for prob, vf in zip(self.distribution, self.state_to_vf):
            if vf.RO < 0:
                assert np.isclose(prob, 0.0, rtol=np.float64(1e-10)), f"{prob=}"
            else:
                result[vf.RO] += prob
        return result

    def dist_of_tail_slots(self) -> np.ndarray:
        result = np.zeros(self.heur_max + 1, dtype=np.float64)
        assert len(self.distribution) == len(self.state_to_vf)
        for prob, vf in zip(self.distribution, self.state_to_vf):
            result[vf.tail_slots] += prob

        return result

    def max(self, other: "SMValuedDistribution") -> "SMValuedDistribution":
        new_distribution = (
            np.cumsum(self.distribution) * other.distribution
            + np.cumsum(other.distribution) * self.distribution
            - self.distribution * other.distribution
        )
        return SMValuedDistribution(
            distribution=new_distribution,
            state_to_vf=self.state_to_vf,
            vf_to_index=self.vf_to_index,
            heur_max=self.heur_max,
        )

    def make_binom(
        alpha: np.float64,
        tail_slots_to_value: dict[int, np.float64],
        sacrifice: int,
        heur_max: int = SLOTS_PER_EPOCH,
    ) -> "SMValuedDistribution":
        """
        chances[tail_slots, slots] = P(tail_slots=EpochString.tail_slots and slots=EpochString.slots=slots)
        if EpochString ~ EpochStrings {"ahaa...a", "aah...ah"...}
        """
        tail_slots: np.ndarray
        slots: np.ndarray
        tail_slots, slots = np.indices((SLOTS_PER_EPOCH + 1, SLOTS_PER_EPOCH + 1))

        chances_full = (
            binom.pmf(
                slots - tail_slots,
                SLOTS_PER_EPOCH - tail_slots - (tail_slots < SLOTS_PER_EPOCH),
                alpha,
            )
            * alpha**tail_slots
            * (1 - alpha) ** (tail_slots < SLOTS_PER_EPOCH)
        )
        chances = np.vstack(
            (chances_full[:heur_max], np.sum(chances_full[heur_max:], axis=0))
        )
        distribution = np.zeros(
            (heur_max + 1) * (2 * SLOTS_PER_EPOCH + 1), dtype=np.float64
        )

        state_to_vf: list[SMEpochString] = [
            SMEpochString(
                tail_slots=t,
                RO=s - sacrifice,
                value=tail_slots_to_value[t] + s - sacrifice,
            )
            for t in range(0, heur_max + 1)
            for s in range(0, SLOTS_PER_EPOCH + 1)
        ]
        for t in range(0, heur_max + 1):
            for RO in range(-SLOTS_PER_EPOCH, SLOTS_PER_EPOCH + 1):
                vf = SMEpochString(
                    tail_slots=t, RO=RO, value=tail_slots_to_value[t] + RO
                )
                if vf not in state_to_vf:
                    state_to_vf.append(vf)

        state_to_vf = sorted(state_to_vf, key=lambda x: x.value)
        for i, vf in enumerate(state_to_vf):
            slots = vf.RO + sacrifice
            if 0 <= slots <= SLOTS_PER_EPOCH:
                distribution[i] = chances[vf.tail_slots, slots]

        vf_to_index: dict[SMEpochString, int] = {
            sm_es: i for i, sm_es in enumerate(state_to_vf)
        }
        return SMValuedDistribution(
            distribution=distribution,
            state_to_vf=state_to_vf,
            vf_to_index=vf_to_index,
            heur_max=heur_max,
        )


def iterate_value_function(
    alpha: np.float64, value_function: dict[int, np.float64]
) -> tuple[dict[int, np.float64], dict[int, SMValuedDistribution]]:
    distribution: SMValuedDistribution = SMValuedDistribution.make_binom(
        alpha=alpha,
        tail_slots_to_value=value_function,
        sacrifice=0,
        heur_max=len(value_function) - 1,
    )
    tail_slots_to_dist: dict[int, SMValuedDistribution] = {}

    new_value_function: dict[int, np.float64] = {}
    for tail_slots in range(distribution.heur_max + 1):
        new_value_function[tail_slots] = distribution.expected_value()
        tail_slots_to_dist[tail_slots] = deepcopy(distribution)

        if tail_slots != distribution.heur_max + 1:
            distribution = distribution.max(other=distribution.increment_sacrifice())

    return new_value_function, tail_slots_to_dist


def solve_sm_approximating_optimal(
    alpha: np.float64, num_of_iteration: int, heur_max: int
) -> np.ndarray:
    vf: dict[int, np.float64] = {i: np.float64(0.0) for i in range(0, heur_max + 1)}
    tail_slots_to_dist: Optional[dict[int, SMValuedDistribution]] = None
    for _ in range(num_of_iteration):
        vf, tail_slots_to_dist = iterate_value_function(alpha=alpha, value_function=vf)

    assert tail_slots_to_dist is not None
    state_transition_l: list[np.ndarray[np.float64]] = []

    for t in range(heur_max + 1):
        state_transition_l.append(tail_slots_to_dist[t].dist_of_tail_slots())

    state_trainsition_matrix = np.stack(state_transition_l)

    n = heur_max + 1
    A = state_trainsition_matrix - np.eye(n)
    A = np.vstack([A.T, np.ones(n)])
    b = np.zeros(n)
    b = np.append(b, 1)

    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    cashout_matrix = np.zeros((SLOTS_PER_EPOCH + 1, n), dtype=np.float64)
    for t in range(heur_max + 1):
        cashout_matrix[:, t] = tail_slots_to_dist[t].dist_of_RO()

    distribution_of_RO = np.matmul(cashout_matrix, pi)

    return distribution_of_RO


def selfish_mixing(alphas: list[np.float64], iterations: int, heur_max: int):
    cache_path = os.path.join(CACHE_FOLDER, f"selfish_mixing_{heur_max}.json")
    result: dict[np.float64, np.float64] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            result = {
                np.float64(key): np.float64(val) for key, val in json.load(f).items()
            }

    alphas = [alpha for alpha in alphas if alpha not in result]
    for alpha in tqdm(alphas, desc="selfish mixing"):
        dist = solve_sm_approximating_optimal(
            alpha=alpha, num_of_iteration=iterations, heur_max=heur_max
        )
        result[alpha] = np.dot(dist, np.arange(SLOTS_PER_EPOCH + 1))

    with open(cache_path, "w") as f:
        json.dump(result, f)
