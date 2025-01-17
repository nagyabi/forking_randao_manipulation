from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from base.helpers import SLOTS_PER_EPOCH
from theory.method.detailed_distribution import DetailedSlot, Outcome
from theory.method.engine import BeaconChainAction


@dataclass
class QuantizedEAS(ABC):
    slot: int
    id_to_outcome: dict[int, Outcome]
    num_of_epoch_strings: int
    epoch_string_to_index: dict[str, int]

    @abstractmethod
    def run(self, id_to_epoch: dict[int, tuple[int, str]]) -> tuple[int, list[int]]:
        pass

    @abstractmethod
    def run_prereq(self) -> list[int]:
        pass

    @abstractmethod
    def run_after(
        self, id_to_epoch: dict[int, tuple[int, str]], normalized_id: int
    ) -> int:
        pass

    @abstractmethod
    def run_after_prereq(self, normalized_id: int) -> list[int]:
        pass

    def _run_quant_group(
        self,
        id_to_epoch: dict[int, tuple[int, str]],
        group: dict[int, list[int]],
    ) -> int:
        """
        Returns the id of the 'best' RO, epoch_string tuple.

        Args:
            id_to_epoch (dict[int, tuple[int, str]]): id to (RO, epoch_string) values in epoch E+2
            group (dict[int, list[int]]): quantized group

        Returns:
            int: id in id_to_epoch which gives maximal result in the quantized model
        """
        best, best_val = None, -1
        for id, scores in group.items():
            index = 0
            if id in id_to_epoch:
                adv_slots, epoch_string = id_to_epoch[id]
                assert self.id_to_outcome[id].known
                index = (
                    adv_slots * self.num_of_epoch_strings
                    + self.epoch_string_to_index[epoch_string]
                )
            else:
                assert not self.id_to_outcome[
                    id
                ].known, f"{id=} {list(group)=} {self.id_to_outcome=}"

            if best_val < scores[index]:
                best_val = scores[index]
                best = id

        assert best_val != -1
        return best


@dataclass
class SMQuantizedEAS(QuantizedEAS):
    group: dict[int, list[int]]

    def run(self, id_to_epoch: dict[int, tuple[int, str]]) -> tuple[int, list[int]]:
        return (
            self._run_quant_group(
                id_to_epoch=id_to_epoch,
                group=self.group,
            ),
            [],
        )

    def run_prereq(self) -> list[int]:
        return list(self.id_to_outcome)

    def run_after(
        self, id_to_epoch: dict[int, tuple[int, str]], normalized_id: int
    ) -> int:
        return self.run(id_to_epoch)[0]

    def run_after_prereq(self, normalized_id: int) -> list[int]:
        return self.run_prereq()


@dataclass
class ForkQuantizedEAS(QuantizedEAS):
    head_group: dict[int, list[int]]
    regret_groups: list[dict[int, list[int]]]

    def run(self, id_to_epoch: dict[int, tuple[int, str]]) -> tuple[int, list[int]]:
        best_id = self._run_quant_group(
            id_to_epoch=id_to_epoch,
            group=self.head_group,
        )
        correct_groups: list[dict[int, list[int]]] = [
            group for group in self.regret_groups if best_id in group
        ]
        known = self.id_to_outcome[best_id].known
        assert (
            not known or len(correct_groups) == 1
        ), f"No matching/unambigous group for {best_id=} {correct_groups=}"
        return best_id, (
            [id for id in correct_groups[0] if self.id_to_outcome[id].known]
            if known
            else []
        )

    def run_prereq(self) -> list[int]:
        return [id for id in self.head_group if self.id_to_outcome[id].known]

    def run_after(
        self, id_to_epoch: dict[int, tuple[int, str]], normalized_id: int
    ) -> int:
        correct_groups: list[dict[int, list[int]]] = [
            group for group in self.regret_groups if normalized_id in group
        ]
        assert (
            len(correct_groups) == 1
        ), f"No matching/unambigous group for {normalized_id=}"
        return self._run_quant_group(id_to_epoch=id_to_epoch, group=correct_groups[0])

    def run_after_prereq(self, normalized_id: int) -> list[int]:
        correct_groups: list[dict[int, list[int]]] = [
            group for group in self.regret_groups if normalized_id in group
        ]
        assert (
            len(correct_groups) == 1
        ), f"No matching/unambigous group for {normalized_id=}"
        return [id for id in correct_groups[0] if self.id_to_outcome[id].known]


class RANDAODataProvider(ABC):
    def __init__(self):
        self.provided: dict[int, tuple[int, str]] = {}

    @abstractmethod
    def _provide(self, cfg: int, **kwargs) -> tuple[int, str]:
        """
        Args:
            cfg (int): number encoding CANONICAL/NONCANONICAL values with 1/0 digits.
            The least significant bit corresponds to the first character of the attack
            string (or eas). Exactly len(attack_string.split(".")[0]) must be considered.
            In practice an actual RANDAO value can be computed from it.
        Returns:
            tuple[int, str]: The first element is the adversarial slots in epoch E+2
            for the given attacker, the second one is the epoch string. For example
            (12, "ha#aahaa") is returned if epoch e+2 looks like this: haaahhah...aahaa
            The epoch string needs to be truncated by size_prefix and size_postfix.
            Additionally, the string before "#" can only contain at most one "a" characters,
            ("", "a", "ha" ... "hh...ha")
        """

    def feed_result(self, cfg: int) -> None:
        """
        Feedback for the agent about the final config (cfg) of the attack.
        """

    def feed_actions(self, actions: list[BeaconChainAction]) -> None:
        """
        Feedback to the agent about the recommended move according to the quantized model.
        """

    def provide(self, cfg: int) -> tuple[int, str]:
        if cfg not in self.provided:
            self.provided[cfg] = self._provide(cfg=cfg)
        return self.provided[cfg]


class RandomRANDAODataProvider(RANDAODataProvider):
    def __init__(
        self,
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
        generator: np.random.Generator,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix
        self.generator = generator
        self.generated: dict[int, tuple[int, str]] = {}

    def _provide(self, cfg: int) -> tuple[int, str]:
        sample = np.random.choice(
            [0, 1], size=SLOTS_PER_EPOCH, p=[1 - self.alpha, self.alpha]
        )
        adv_slots = np.sum(sample)
        prefix = "".join("a" if s == 1 else "h" for s in sample[: self.size_prefix])
        if prefix.count("a") > 0:
            p, _ = prefix.split("a", 1)
            prefix = p + "a"
        else:
            prefix = ""
        postfix = "".join("a" if s == 1 else "h" for s in sample[-self.size_postfix :])
        while len(postfix) > 0 and postfix[0] == "h":
            postfix = postfix[1:]
        return adv_slots, f"{prefix}#{postfix}"


class RandomSimpleRANDAODataProvider(RANDAODataProvider):
    """
    Randomly generates an (RO, ES), where ES is "#" and RO ~ Binom(32, alpha)
    """

    def __init__(
        self,
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
    ):
        super().__init__()
        self.alpha = alpha
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix

    def _provide(self, cfg: int, **kwargs) -> tuple[int, str]:
        res = np.random.binomial(SLOTS_PER_EPOCH, self.alpha)
        while res > SLOTS_PER_EPOCH - self.size_postfix - self.size_prefix:
            res = np.random.binomial(SLOTS_PER_EPOCH, self.alpha)
        return res, "#"
