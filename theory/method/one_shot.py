from dataclasses import dataclass

import numpy as np

from theory.method.distribution import DistMakerBase
from theory.method.evaluate import Evaluator
from theory.method.preeval import (
    calc_attack_string_mapping,
    calc_eas_mapping,
    calc_postfix_mapping,
)


@dataclass(frozen=True)
class OneShotStat:
    known: int  # number of known outcomes ahead
    unknown: int  # number of unknown outcomes ahead

    def expected_value_in_distribution(self) -> "OneShotStat":
        return OneShotStat(
            known=0,
            unknown=self.known + self.unknown,
        )

    def increase_sacrifice(self, sacrifice: int) -> "OneShotStat":
        return self

    def max(self, other: "OneShotStat") -> "OneShotStat":
        return OneShotStat(
            known=self.known + other.known, unknown=max(self.unknown, other.unknown)
        )

    def max_unknown(self, unknown: "OneShotStat") -> "OneShotStat":
        return OneShotStat(
            known=self.known, unknown=unknown.known + max(self.unknown, unknown.unknown)
        )


class OneShotMaker(DistMakerBase[OneShotStat]):
    def make_distribution(self, **kwargs) -> OneShotStat:
        return OneShotStat(known=1, unknown=0)


@dataclass
class OneShotCalculator:
    size_prefix: int
    size_postfix: int

    def eval_all_eas(
        self,
        alpha: np.float64,
    ) -> dict[str, OneShotStat]:
        known, attack_string_mapping = calc_attack_string_mapping(
            alpha=alpha,
            size_prefix=self.size_prefix,
            size_postfix=self.size_postfix,
        )
        eas_mapping = calc_eas_mapping(
            attack_string_mapping=attack_string_mapping,
            postfix_mapping=calc_postfix_mapping(attack_string_mapping),
        )
        evaluator = Evaluator(
            alpha=alpha,
            size_prefix=self.size_prefix,
            size_postfix=self.size_postfix,
            known=known,
            eas_mapping=eas_mapping,
            optimize=False,
        )
        result: dict[str, OneShotStat] = evaluator.eval_all_eas(
            None, OneShotMaker(), desc=f"oneshot {alpha}"
        )
        return {eas: st.known + st.unknown for eas, st in result.items()}

    def calc_prob(self, alpha: np.float64) -> float:
        eas_to_choices = self.eval_all_eas(alpha=alpha)
        attack_string_to_choices = {
            eas.split("#")[0]: choice
            for eas, choice in eas_to_choices.items()
            if eas.split("#")[1] == ""
        }
        result = 0.0
        pr_sum = 0.0
        for attack_string, choices in attack_string_to_choices.items():

            before, after = attack_string.split(".")
            num_of_as = before.count("a")
            prob = alpha ** (num_of_as) * (1 - alpha) ** (self.size_postfix - num_of_as)
            if after.count("a") > 0:
                prob *= alpha * (1 - alpha) ** after.count("h")
            else:
                prob *= (1 - alpha) ** self.size_prefix
            result += prob * (1 - (1 - alpha) ** choices)
            pr_sum += prob
        assert np.isclose(pr_sum, 1.0, rtol=1e-7), f"{pr_sum=}"
        assert 0 <= result <= 1, f"{result=} {alpha=}"
        return result
