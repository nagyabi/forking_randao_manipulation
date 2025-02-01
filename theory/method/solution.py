from typing import Any, Optional
import numpy as np
from theory.method.utils.attack_strings import ExtendedAttackString
from theory.method.cache import Cacher
from theory.method.detailed_distribution import DetailedDistribution
from theory.method.detailed_eval import DetailedEvaluator
from theory.method.distribution import (
    ApproximatedDistributionMaker,
    DetailedDistributionMaker,
    RichDistributionMaker,
    RichValuedDistribution,
    ValuedDistribution,
    ValuedDistributionMaker,
)
from theory.method.evaluate import Evaluator
from scipy.sparse import lil_matrix

from base.beaconchain import AttackStringEntry
from base.helpers import SLOTS_PER_EPOCH
from theory.method.preeval import (
    calc_attack_string_mapping,
    calc_eas_mapping,
    calc_postfix_mapping,
)
from theory.method.quant.quantize import Quantizer

index_to_name = {
    1: "slots",
    2: "forked_honest_blocks",
    3: "fork_prob",
    4: "regret_prob",
    5: "sm_prob",
    6: "exp_fork",
    7: "exp_regret",
    8: "exp_sm",
}


class NewMethodSolver:
    def __init__(
        self,
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
        cache_folder: Optional[str] = None,
    ):
        self.alpha = alpha
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix
        default = {
            str(eas): np.float64(0.0)
            for eas in ExtendedAttackString.possibilities(
                size_prefix=size_prefix, size_postfix=size_postfix
            )
        }
        self.cacher = Cacher(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            default=default,
            base_folder=cache_folder,
        )
        self.known, attack_string_mapping = calc_attack_string_mapping(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
        )
        self.eas_mapping = calc_eas_mapping(
            attack_string_mapping=attack_string_mapping,
            postfix_mapping=calc_postfix_mapping(attack_string_mapping),
        )
        self.maker = ApproximatedDistributionMaker(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            eas_mapping=self.eas_mapping,
            memory_size=65,
        )
        self.rich_maker = RichDistributionMaker(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            eas_mapping=self.eas_mapping,
        )
        self.detailed_maker = DetailedDistributionMaker(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            eas_mapping=self.eas_mapping,
        )
        self.evaluator = Evaluator(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            known=self.known,
            eas_mapping=self.eas_mapping,
        )
        quantizer = Quantizer(
            size_prefix=self.size_prefix,
            size_postfix=self.size_postfix,
            eas_mapping=self.eas_mapping,
        )
        self.detailed_evaluator = DetailedEvaluator[DetailedDistribution](
            maker=self.detailed_maker,
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            known=self.known,
            eas_mapping=self.eas_mapping,
            quantizer=quantizer,
        )

    def finetune_value_function(self, num_of_iterations: int) -> dict[str, np.float64]:
        iteration, vf = self.cacher.value_function(iteration=num_of_iterations)
        while iteration <= num_of_iterations:
            eas_to_dist: dict[str, ValuedDistribution] = self.evaluator.eval_all_eas(
                value_function=vf, maker=self.maker, desc=f"{iteration=}"
            )
            vf = {eas: dist.expected_value() for eas, dist in eas_to_dist.items()}
            self.cacher.push_value_function(iter=iteration, value_function=vf)
            iteration += 1
        return vf

    def eval_value_function_rich(
        self, value_function: dict[str, np.float64], iteration_num: int
    ) -> tuple[dict[str, list[str]], dict[str, RichValuedDistribution]]:
        maybe_distribution = self.cacher.rich_distributions(iteration=iteration_num)
        if maybe_distribution is not None:
            return maybe_distribution
        distributions: dict[str, RichValuedDistribution] = self.evaluator.eval_all_eas(
            value_function=value_function, maker=self.rich_maker, desc="final iteration"
        )
        distributions = {
            eas: distribution.expected_value_in_distribution()
            for eas, distribution in distributions.items()
        }
        self.cacher.push_rich_distributions(
            meanings=self.rich_maker.postfix_to_meanings,
            r_dists=distributions,
            iteration=iteration_num,
        )
        return self.rich_maker.postfix_to_meanings, distributions

    def eval_value_function_detailed(
        self, value_function: dict[str, np.float64], iteration_num: int
    ) -> None:
        if not self.cacher.already_quant(iteration=iteration_num):
            self.detailed_evaluator.eval_all_eas(
                value_function=value_function, desc="detailed iteration"
            )
            self.cacher.push_quant(
                self.detailed_evaluator.quantizer, iteration=iteration_num
            )

        # self.cacher.push_detailed_distribution(d_dists=distributions, iteration=iteration_num)
        # return distributions

    def markov_chain(
        self,
        rich_distributions: dict[str, RichValuedDistribution],
        meanings: dict[str, list[str]],
        eas_mapping: dict[str, str],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
        """
        Construction of the markov chain and the cashout vector

        Args:
            rich_distributions (dict[str, RichValuedDistribution]): Evaluated rich distributions

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The first member is the state transition matrix,
            the second, third and forth are vectors with the corresponding expected values of ROs, adversarial slots
            forked honest slots. The last is the states of the matrix.
        """

        states_of_markov_chain = list(
            set(eas_mapping.values())
        )  # EASs, we need to fix an order
        eas_to_index: dict[str, int] = {
            eas: index for index, eas in enumerate(states_of_markov_chain)
        }

        matrix = lil_matrix(
            (len(states_of_markov_chain), len(states_of_markov_chain)), dtype=np.float64
        )

        ROs: list[np.float64] = []

        rest_stats: dict[str, list[np.float64]] = {
            stat_name: [] for stat_name in index_to_name.values()
        }

        for index, state in enumerate(states_of_markov_chain):
            distribution = rich_distributions[state].expected_value_in_distribution()
            ROs.append(distribution.distribution[0].expected_value_of_vars[0])

            for i, name in index_to_name.items():
                rest_stats[name].append(
                    distribution.distribution[0].expected_value_of_vars[i]
                )
            assert np.isclose(
                np.sum(distribution.distribution[0].distribution_of_epoch_strings),
                1.0,
                rtol=1e-10,
            ), f"{np.sum(distribution.distribution[0].distribution_of_epoch_strings)=}"

            postfix = state.split("#")[1]
            assert len(meanings[postfix]) == len(
                distribution.distribution[0].distribution_of_epoch_strings
            )
            for next_state, probability in zip(
                meanings[postfix],
                distribution.distribution[0].distribution_of_epoch_strings,
            ):

                matrix[eas_to_index[next_state], index] = probability

        ROs = np.array(ROs, dtype=np.float64)
        return (
            matrix,
            ROs,
            {
                stat_name: np.array(stat, dtype=np.float64)
                for stat_name, stat in rest_stats.items()
            },
            states_of_markov_chain,
        )

    def solve_markov_chain(
        self, matrix: np.ndarray, ROs: np.ndarray, epsilon: np.float64
    ) -> tuple[np.ndarray, np.float64]:
        n = int(ROs.shape[0])

        b = np.zeros(n, dtype=np.float64)
        b[0] = 1.0

        # b = np.append(b, 1)
        print("Solving the matrix")
        A = matrix.tocsr()

        RO_prev = None
        RO = None

        for _ in range(10):  # Initial
            RO_prev = RO
            RO = np.dot(b, ROs)
            b = A.dot(b)
            b /= np.sum(b)

        while not np.isclose(RO, RO_prev, rtol=epsilon):
            RO_prev = RO
            RO = np.dot(b, ROs)
            b = A.dot(b)
            b /= np.sum(b)

        for _ in range(100):  # Further refinement
            RO = np.dot(b, ROs)
            b = A.dot(b)
            b /= np.sum(b)
        return b, RO

    def attack_string_contributions(
        self,
        pi: np.ndarray,
        ROs: np.ndarray,
        res_stats: dict[str, np.ndarray],
        EASs: list[str],
    ) -> dict[str, AttackStringEntry]:
        result: dict[str, dict[str, Any]] = {}
        assert len(pi) == len(EASs)
        for i, (eas, RO, prob) in enumerate(zip(EASs, ROs, pi)):
            attack_string = eas.split("#")[0]
            if attack_string not in result:
                result[attack_string] = {
                    "RO": {
                        "value": np.float64(0.0),
                        "scaled": np.float64(0.0),
                    },
                    **{
                        stat_name: {
                            "value": np.float64(0.0),
                            "scaled": np.float64(0.0),
                        }
                        for stat_name in res_stats
                    },
                    "probability": np.float64(0.0),
                }
            result[attack_string]["probability"] += prob
            result[attack_string]["RO"]["scaled"] += prob * RO
            result[attack_string]["RO"]["value"] = (
                result[attack_string]["RO"]["scaled"]
                / result[attack_string]["probability"]
            )
            for stat_name, stat in res_stats.items():
                result[attack_string][stat_name]["scaled"] += prob * stat[i]
                result[attack_string][stat_name]["value"] = (
                    result[attack_string][stat_name]["scaled"]
                    / result[attack_string]["probability"]
                )
        result = {
            key: val
            for key, val in sorted(
                result.items(), key=lambda x: x[1]["probability"], reverse=True
            )
        }
        return result

    def solve(
        self,
        num_of_iterations: int,
        markov_chain: bool,
        quant: bool,
        epsilon: np.float64 = np.float64(1e-20),
    ) -> np.float64:
        vf = self.finetune_value_function(num_of_iterations=num_of_iterations)
        if quant:
            self.eval_value_function_detailed(
                value_function=vf, iteration_num=num_of_iterations
            )
        if not markov_chain:
            return np.float64(0.0)  # TODO
        meanings, rds = self.eval_value_function_rich(
            value_function=vf, iteration_num=num_of_iterations
        )
        matrix, ROs, res_stats, EASs = self.markov_chain(
            rich_distributions=rds,
            meanings=meanings,
            eas_mapping=self.evaluator.eas_mapping,
        )
        b, RO = self.solve_markov_chain(matrix=matrix, ROs=ROs, epsilon=epsilon)

        expected_values = self.cacher.expected_values()
        expected_values[num_of_iterations] = {
            "RO": RO,
            **{stat_name: np.dot(b, stat) for stat_name, stat in res_stats.items()},
        }
        self.cacher.push_expected_values(expected_values=expected_values)
        contrib = self.attack_string_contributions(
            pi=b,
            ROs=ROs,
            res_stats=res_stats,
            EASs=EASs,
        )
        self.cacher.push_attack_strings_to_probabilities(contrib, num_of_iterations)
        return RO


if __name__ == "__main__":
    i = 0
    while True:
        alpha = np.float64(0.001 + i * 0.01)
        print(f"Calculating for {alpha=}")

        if alpha > 0.45:
            break
        solver = NewMethodSolver(alpha=alpha, size_prefix=2, size_postfix=6)
        RO = solver.solve(num_of_iterations=10)
        print(
            f"Stake: {round(alpha * 100, 5)}% => {round(100 * RO / SLOTS_PER_EPOCH, 5)}%"
        )
        i += 1
