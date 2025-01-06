import math

import numpy as np
from base.helpers import BIG_ENTITIES, LATEST_DELIVERY, SLOTS_PER_EPOCH
from base.statistics import read_delivery_cases, to_selfish_mixing
from scipy.special import gamma

import scipy

MIN_CASES = 5


def normalize_proposed(
    delivery_path: str, entities: list[str]
) -> dict[str, list[float]]:
    rows = read_delivery_cases(delivery_path)
    rows = sorted(rows, key=lambda x: x.epoch)

    rows = to_selfish_mixing(rows)
    rows = [
        row
        for row in rows
        if row.real_statuses.count("m") == 0 and row.real_statuses.count("r") == 0
    ]

    result: dict[str, list[float]] = {}
    for entity in entities:
        filtered = [row for row in rows if row.attacker == entity]
        assert len(set(row.epoch for row in filtered)) == len(filtered)

        normalized = [
            (row.RO - SLOTS_PER_EPOCH * row.stake + row.real_statuses.count("m"))
            / math.sqrt(SLOTS_PER_EPOCH * row.stake * (1 - row.stake))
            for row in filtered
        ]
        result[entity] = normalized

    return result


def observed_vs_expected(
    delivery_path: str, entities: list[str], proposed: bool
) -> dict[str, tuple]:
    rows = read_delivery_cases(delivery_path)
    rows = sorted(rows, key=lambda x: x.epoch)

    rows = to_selfish_mixing(rows)
    rows = [
        row
        for row in rows
        if ((row.real_statuses.count("m") != 0) ^ proposed)
        and row.real_statuses.count("r") == 0
    ]

    result: dict[str, tuple] = {}
    for entity in entities:
        filtered = [row for row in rows if row.attacker == entity]
        # print(f"{entity=} -> {len(filtered)}")
        assert len(set(row.epoch for row in filtered)) == len(filtered)
        bin_edges = np.arange(0, SLOTS_PER_EPOCH + 2)

        data = np.array(
            [entry.RO + entry.real_statuses.count("m") for entry in filtered],
            dtype=np.float64,
        )
        hist = np.histogram(data, bins=bin_edges)[0]
        # print(hist)
        stakes = np.array([entry.stake for entry in filtered], dtype=np.float64)
        expected_freqs = np.array(
            [
                np.average(scipy.stats.binom.pmf(k, SLOTS_PER_EPOCH, stakes))
                for k in range(SLOTS_PER_EPOCH + 1)
            ],
            dtype=np.float64,
        )
        result[entity] = (hist, expected_freqs)
    return result


def observed_vs_expected_interpol(
    delivery_path: str, entities: list[str], proposed: bool, density: float
) -> dict[str, tuple]:
    rows = read_delivery_cases(delivery_path)
    rows = sorted(rows, key=lambda x: x.epoch)

    rows = to_selfish_mixing(rows)
    rows = [
        row
        for row in rows
        if ((row.real_statuses.count("m") != 0) ^ proposed)
        and row.real_statuses.count("r") == 0
    ]

    result: dict[str, tuple] = {}
    for entity in entities:
        filtered = [row for row in rows if row.attacker == entity]

        assert len(set(row.epoch for row in filtered)) == len(filtered)
        bin_edges = np.arange(0, SLOTS_PER_EPOCH + 2)

        data = np.array(
            [entry.RO + entry.real_statuses.count("m") for entry in filtered],
            dtype=np.float64,
        )
        hist = np.histogram(data, bins=bin_edges)[0]

        stakes = np.array([entry.stake for entry in filtered], dtype=np.float64)
        elem = -0.5
        exp_xs = []
        while elem < SLOTS_PER_EPOCH:
            exp_xs.append(elem)
            elem += density
        exp_xs = np.array(exp_xs, dtype=np.float64)
        multiplier = (
            gamma(SLOTS_PER_EPOCH + 1)
            / gamma(1 + exp_xs)
            / gamma(SLOTS_PER_EPOCH + 1 - exp_xs)
        )
        exp_ys = [
            multi * np.average(stakes**x * (1 - stakes) ** (SLOTS_PER_EPOCH - x))
            for (x, multi) in zip(exp_xs, multiplier)
        ]

        result[entity] = (hist, (exp_xs, np.array(exp_ys, dtype=np.float64)))
    return result


def group_cases(hist: np.array, exp: np.array) -> tuple[np.array, np.array]:
    h_cum, e_cum = 0, 0.0
    hist_res = []
    exp_res = []
    need_to = True
    for h, e in zip(hist, exp):
        h_cum += h
        e_cum += e
        if h_cum >= MIN_CASES:
            hist_res.append(h_cum)
            exp_res.append(e_cum)
            h_cum, e_cum = 0, 0.0
            need_to = False
        else:
            need_to = True
    if need_to:
        hist_res.append(h_cum)
        exp_res.append(e_cum)
    return np.array(hist_res, dtype=np.int32), np.array(exp_res, dtype=np.float64)


def test_hypothesis(act, exp) -> tuple[int, float]:
    act_n, exp_n = group_cases(act[::-1], exp[::-1])
    act_n, exp_n = group_cases(act_n[::-1], exp_n[::-1])
    _, p_val = scipy.stats.chisquare(act_n, exp_n * np.sum(act_n))
    return len(act_n), p_val


if __name__ == "__main__":
    data = observed_vs_expected(LATEST_DELIVERY, BIG_ENTITIES, proposed=True)
    for entity, (act, exp) in data.items():
        bins, p_val = test_hypothesis(act, exp)
        if bins >= 3 and p_val == p_val:
            print(
                f"{entity} & ${np.sum(act)}$                & ${bins}$                & ${round(p_val, 3)}$                                                       \\\\"
            )
            # print(f"All {entity} => {bins=} p_val={round(p_val, 5)} N={np.sum(act)}")
    # data = observed_vs_expected(LATEST_DELIVERY, BIG_ENTITIES, proposed=False)
    # for entity, (act, exp) in data.items():
    #     bins, p_val = test_hypothesis(act, exp)
    #     if bins >= 3 and p_val == p_val:
    #
    #         print(f"Missed {entity} => {bins=} {p_val=}")
