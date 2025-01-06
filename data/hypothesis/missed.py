from dataclasses import dataclass
import math
import os
import random
from typing import Optional

import numpy as np
from tqdm import tqdm
from base.helpers import BIG_ENTITIES, FIGURES_FOLDER, LATEST_DELIVERY, SLOTS_PER_EPOCH
from base.statistics import CasesEntry, read_delivery_cases, to_selfish_mixing
from scipy.stats import norm

from make_figures.base import plot_data


"""
In this file we test whether missing slots near the end of the epoch was related to RANDAO manipulation
H0: The examined entity only missed by accident
Testing: We take a look at the slots in epoch E+2: X_1, X_2, ... X_N
If H0 holds: X_n ~ Binom(32, p_n), thus X_1 + X_2 + ... X_N ~ Poisson-Binomial(p_1 32 times, ... p_N)
We can approximate this if N > 50 with a N(32 * (X_1 + X_2 + ...), N * (p_1 * (1 - p_1) + p2 * (1 - p_2) + ...)) (https://en.wikipedia.org/wiki/Poisson_binomial_distribution)
"""

MIN_CASES = 50
SIM_SIZE = 10000


@dataclass
class StatTestResult:
    p_value: float
    avg: float
    summed: float
    expected: float
    sqrt_var: float
    N: int


def test_missed_slots(
    delivery: list[CasesEntry], entity: str
) -> Optional[StatTestResult]:
    """
    Tests H0: the relevant missed slots were by accident.

    Args:
        delivery (list[CasesEntry]): latest delivery
        entitiy (str): entity to test H0 for

    Returns:
        Optional[tuple[float, float, int]]: None if tgere aren't sufficient cases.
        Otherwise the first element is the p-value of the element from the normalized value,
        while the second is the average of the stakes where the entity missed.
        The third value is the sample size.
    """

    filtered = [entry for entry in delivery if entry.attacker == entity]
    if len(filtered) < MIN_CASES:
        return None
    summed = sum(entry.RO + entry.real_statuses.count("m") for entry in filtered)
    exp = SLOTS_PER_EPOCH * sum(entry.stake for entry in filtered)
    variance = SLOTS_PER_EPOCH * sum(
        entry.stake * (1 - entry.stake) for entry in filtered
    )
    avg = sum(entry.stake for entry in filtered) / len(filtered)
    normalized = (summed - exp) / math.sqrt(variance)

    # print(f"{normalized=}")
    p_value = 1 - norm.cdf(normalized)
    return StatTestResult(
        p_value=p_value,
        avg=avg,
        summed=summed,
        expected=exp,
        sqrt_var=math.sqrt(variance),
        N=len(filtered),
    )


def test_entities(delivery_path: str, entities: list[str]):
    delivery = read_delivery_cases(delivery_path)
    delivery = to_selfish_mixing(delivery)
    contains_missed = [
        entry
        for entry in delivery
        if entry.real_statuses.count("m") > 0 and entry.real_statuses.count("r") == 0
    ]
    for entity in entities:
        result = test_missed_slots(delivery=contains_missed, entity=entity)
        if result is None:
            print(f"{entity} has not enough data, skipping..")
        else:
            print(
                f"For {entity} => p-value={result.p_value} avg(alphas)={result.avg} mu={result.summed} mu_0={result.expected}, sigma_0={result.sqrt_var} N={result.N}"
            )


def generate_manipulated_data(
    delivery: list[CasesEntry], entity: str, chance_of_man: float
) -> list[CasesEntry]:
    """
    Generating synthetic data from a theoretical entitiy if he would have manipulated the RANDAO
    We generate until the chosen epoch contains a missed slot, so the generated data has the same bias
    (We want the RO | there is at least one missed slot)

    Args:
        delivery (list[CasesEntry]): delivery to transform
        entity (str): entity to generate synthetic data for
        chance_of_man (float): chance of manipulating, the rest comes from network outage

    Returns:
        list[CasesEntry]: Generated delivery
    """
    delivery = [entry for entry in delivery if entry.attacker == entity]

    result: list[CasesEntry] = []
    for entry in delivery:
        assert entry.attack_string.count("h") == 0
        RO: int
        real_statuses: str | None = None
        n = len(entry.attack_string.split(".")[0])
        assert n > 0
        if np.random.uniform(0, 1) < chance_of_man:
            # No network outage, we try to manipulate the RANDAO
            while real_statuses is None or real_statuses.count("m") == 0:
                stat_to_RO: dict[str, int] = {}
                for cfg in range(2**n):
                    bin_cfg = bin(cfg)[2:]
                    statuses = bin_cfg.replace("0", "m").replace("1", "p")
                    stat_to_RO[statuses] = np.random.binomial(
                        SLOTS_PER_EPOCH, entry.stake
                    ) - statuses.count("m")
                sample = list(stat_to_RO.items())
                random.shuffle(sample)
                real_statuses, RO = max(sample, key=lambda x: x[1])
        else:
            RO = (
                np.random.binomial(SLOTS_PER_EPOCH, p=entry.stake) - 1
            )  # Missing 1 slot
            real_statuses = "p" * n + "m"

        result.append(
            CasesEntry(
                outcome=entry.outcome,  # Dummy data, will not be used
                attacker=entry.attacker,
                stake=entry.stake,
                epoch=entry.stake,
                attack_string=entry.attack_string,
                real_statuses=real_statuses,
                RO=RO,
                statuses_to_RO=None,  # Should crash if it was used during testing
            )
        )
    return result


def test_precision(
    delivery_path: str, entities: str, p_manip: list[float], p_val: float = 0.05
) -> dict[str, dict[float, float]]:
    delivery = read_delivery_cases(delivery_path)
    delivery = to_selfish_mixing(delivery)
    contains_missed = [
        entry
        for entry in delivery
        if entry.real_statuses.count("m") > 0 and entry.real_statuses.count("r") == 0
    ]
    result: dict[str, dict[float, float]] = {}
    for entity in entities:
        filtered = [entry for entry in contains_missed if entry.attacker == entity]
        if len(filtered) < MIN_CASES:
            continue
        result[entity] = {}

        for p_man in p_manip:
            flagged = 0
            for _ in tqdm(range(SIM_SIZE), desc=f"{entity} {p_man}"):
                generated = generate_manipulated_data(
                    delivery=filtered, entity=entity, chance_of_man=p_man
                )
                p = test_missed_slots(delivery=generated, entity=entity)[0]
                if p < p_val:
                    flagged += 1
            result[entity][p_man] = flagged / SIM_SIZE
    return result


def plot_precision(prec: dict[str, dict[float, float]]):
    prec = {
        entity: {float(key): val for key, val in data.items()}
        for entity, data in prec.items()
    }
    plot_data(
        id_to_mapping=prec,
        id_to_name={entity: f"catching {entity} accuraccy" for entity in prec},
        id_to_color={"Lido": "red", "Coinbase": "blue"},
        id_to_linestyle={"Lido": "-", "Coinbase": "-."},
        title="Accuraccy of our test",
        to_filename=os.path.join(FIGURES_FOLDER, "test_accuraccy.png"),
        x_label="Probability of manipulating in each epoch",
        left_y_label="Probability of getting caught",
        right_y_label=None,
    )


if __name__ == "__main__":
    test_entities(LATEST_DELIVERY, entities=BIG_ENTITIES)
    # prec = test_precision(LATEST_DELIVERY, BIG_ENTITIES, p_manip=[i * 0.05 for i in range(1, 21)], p_val=0.05)
