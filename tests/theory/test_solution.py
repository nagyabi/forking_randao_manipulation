import json
import os
import pickle
import tempfile
import numpy as np
import pytest

from base.helpers import sac
from theory.method.detailed_distribution import Action, Move
from theory.method.distribution import RichValuedDistribution
from theory.method.quant.base import (
    RandomRANDAODataProvider,
    RandomSimpleRANDAODataProvider,
)
from theory.method.quant.runner import QuantizedModelRunner
from theory.method.solution import NewMethodSolver

SIZE_SIM = 100_000
SIZE_SMALL_SIM = 1000


def extract_stats(adv_slots: int, before: str, cfg: int) -> np.ndarray:
    is_fork = False
    sac = False
    is_regret = False

    result = np.array([adv_slots, adv_slots, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    for c, adv_char in zip(bin(cfg)[2:].zfill(len(before))[::-1], before):
        if c == "0":
            if adv_char == "a":
                result[0] -= 1
                sac = True
            elif adv_char == "h":
                result[2] += 1
                is_fork = True
            else:
                raise ValueError(f"{adv_char=}")
        elif c == "1":
            if adv_char == "h" and sac:
                is_regret = True
    if is_fork:
        result[3] = 1.0
        result[6] = result[0]
    elif is_regret:
        result[4] = 1.0
        result[7] = result[0]
    elif sac:
        result[5] = 1.0
        result[8] = result[0]

    return result


def check_actions(actions: list[Action], before: str, cfg: int):
    assert len(actions) == len(before), f"{actions=} {before=}"
    for i, (c, adv_char, action) in enumerate(
        zip(bin(cfg)[2:].zfill(len(before))[::-1], before, actions)
    ):
        assert action.slot == -len(before) + i
        if c == "0":
            if adv_char == "h":
                assert action.move == Move.FORKOUT
            else:
                assert action.move == Move.MISSBLOCK
        else:
            if adv_char == "h":
                assert action.move == Move.HONESTPROPOSE
            else:
                assert action.move in [Move.PROPOSEBLOCK, Move.HIDEBLOCK]


arguments = [
    (
        np.float64(0.381),
        1,
        4,
        2,
        ["ah.a#a", "aha.#ah", "a.#a", "aha.a#ahh", "aaha.#aa"],
    ),
    (np.float64(0.191), 2, 3, 1, ["aah.a#a", "aah.a#", "a.#a", "aaa.#aah", "aa.#aa"]),
    (
        np.float64(0.221),
        2,
        4,
        3,
        ["ah.a#", "aha.#ah", "a.#a", "aha.#", "aaha.#aa", "ahaa.#aah"],
    ),
]


@pytest.mark.parametrize(
    "alpha, size_prefix, size_postfix, iteration_num, eass", arguments
)
def test_solution_quant_consistency_with_markov_chain(
    alpha: np.float64,
    size_prefix: int,
    size_postfix: int,
    iteration_num: int,
    eass: list[str],
    epsilon1: float = 0.03,
    epsilon2: float = 0.005,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        solver = NewMethodSolver(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            cache_folder=tmp_dir,
        )
        RO_rich = solver.solve(
            num_of_iterations=iteration_num, markov_chain=True, quant=True
        )
        richdist_path = os.path.join(
            solver.cacher.rich_distribution_folder, f"rd_{iteration_num}.pkl"
        )
        with open(richdist_path, "rb") as f:
            rich_data = pickle.load(f)

        # meanings: dict[str, list[str]]
        rich_dists: dict[str, RichValuedDistribution]

        _, rich_dists = rich_data

        eas_mapping, eas_to_quant, mapping_by_eas_postf = solver.cacher.get_quant(
            iteration=iteration_num
        )
        runner = QuantizedModelRunner(
            eas_to_quantized_eas=eas_to_quant,
            mapping_by_eas_postf=mapping_by_eas_postf,
            eas_mapping=eas_mapping,
        )

        # TESTING THAT THE GIVEN EASS RETURNS CLOSE STATISTICS WHEN QUANTIZED + SIMULATED TO RICH DISTS RESULTS
        generator = np.random.default_rng(42)
        for eas in eass:
            before = eas.split(".")[0]
            simulated_stats: list[np.ndarray] = []
            for _ in range(SIZE_SIM):
                provider = RandomRANDAODataProvider(
                    alpha=alpha,
                    size_prefix=size_prefix,
                    size_postfix=size_postfix,
                    generator=generator,
                )

                cfg = runner.run_one_epoch(eas=eas, provider=provider)
                check_actions(runner.captured_actions, before, cfg)

                adv_slots, _ = provider.provide(cfg)
                simulated_stats.append(extract_stats(adv_slots, before, cfg))

            sim_res = np.sum(np.vstack(simulated_stats), axis=0) / len(simulated_stats)
            rich_res = rich_dists[eas].distribution[0].expected_value_of_vars
            assert np.all(
                np.isclose(rich_res[:3], sim_res[:3], rtol=epsilon1, atol=epsilon1)
            ), f"{rich_res=} {sim_res=}"
            assert np.all(
                np.isclose(rich_res[3:6], sim_res[3:6], rtol=epsilon2, atol=epsilon2)
            ), f"{rich_res=} {sim_res=}"
            assert np.all(
                np.isclose(rich_res[6:], sim_res[6:], rtol=epsilon1, atol=epsilon1)
            ), f"{rich_res=} {sim_res=}"

        # TESTING LONGTERM ATTACK BY SIMULATION
        eas = ".#"
        ROs = []
        for _ in range(SIZE_SIM):
            provider = RandomRANDAODataProvider(
                alpha=alpha,
                size_prefix=size_prefix,
                size_postfix=size_postfix,
                generator=generator,
            )
            before = eas.split(".")[0]
            if before == "":
                cfg = 0
            else:
                cfg = runner.run_one_epoch(eas=eas, provider=provider)
                check_actions(runner.captured_actions, before, cfg)
            adv_slots, epoch_string = provider.provide(cfg)
            stats = extract_stats(adv_slots=adv_slots, before=before, cfg=cfg)
            ROs.append(stats[0])
            eas = eas_mapping[epoch_string.split("#")[1] + "." + epoch_string]

        RO = sum(ROs) / len(ROs)
        path = os.path.join(
            solver.cacher.folder_name,
            f"attack_strings_to_probability_{iteration_num}.json",
        )
        with open(path, "r") as f:
            more_data = json.load(f)
        assert np.isclose(RO, RO_rich, rtol=epsilon1), f"{RO=} {RO_rich=} {more_data=}"
        # TESTING SM CONSISTENCY
        # We generate some cases for all tail slots, the model should always choose the (RO, ES)
        # with the biggest RO, if ES is fix

        for tail_slots in range(1, size_postfix + 1):
            before = "a" * tail_slots
            eas = f"{before}.#aa"

            for _ in range(SIZE_SMALL_SIM):
                provider = RandomSimpleRANDAODataProvider(
                    alpha=alpha, size_prefix=size_prefix, size_postfix=size_postfix
                )
                cfg = runner.run_one_epoch(eas=eas, provider=provider)
                model_pred = provider.provide(cfg)[0] - sac(before=before, cfg=cfg)
                best = max(
                    provider.provide(c)[0] - sac(before=before, cfg=c)
                    for c in range(2**tail_slots)
                )
                assert (
                    best == model_pred
                ), f"{best=} {model_pred=} {tail_slots=} {cfg=} {provider.provided=}"
