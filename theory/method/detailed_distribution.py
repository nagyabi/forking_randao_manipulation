from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol, TypeVar

import numpy as np


class DetailedDistributionLike(Protocol):
    def insert_canonical(
        self: "T_Detailed_Distribution", **kwargs
    ) -> "T_Detailed_Distribution": ...

    def insert_noncanonical(
        self: "T_Detailed_Distribution", is_adv_slot: bool, **kwargs
    ) -> "T_Detailed_Distribution": ...

    def max(
        self: "T_Detailed_Distribution", other: "T_Detailed_Distribution"
    ) -> "T_Detailed_Distribution": ...

    def max_unknown(
        self: "T_Detailed_Distribution", unknown: "T_Detailed_Distribution"
    ) -> "T_Detailed_Distribution": ...

    def expected_value_in_distribution(
        self: "T_Detailed_Distribution",
    ) -> "T_Detailed_Distribution": ...

    def reset(
        self: "T_Detailed_Distribution", regret: bool
    ) -> "T_Detailed_Distribution": ...


T_Detailed_Distribution = TypeVar(
    "T_Detailed_Distribution", bound=DetailedDistributionLike
)


def max_detailed(
    distributions: list[T_Detailed_Distribution],
) -> T_Detailed_Distribution:
    assert len(distributions) > 0
    result = distributions[0]
    for dist in distributions[1:]:
        result = result.max(dist)
    return result


class Move(str, Enum):
    HIDEBLOCK = "HIDEBLOCK"
    MISSBLOCK = "MISSBLOCK"
    PROPOSEBLOCK = "PROPOSEBLOCK"
    FORKOUT = "FORKOUT"
    HONESTPROPOSE = "HONESTPROPOSE"


@dataclass(frozen=True)
class Action:
    slot: int
    move: Move


@dataclass(frozen=True)
class Outcome:
    config: int
    full_config: int
    end_slot: int  # Slot where the suggestions of the best looking outcome should be questioned
    sacrifice: int
    known: bool
    regret: bool
    relevant_actions: tuple[Action, ...]

    def from_dict(dct: dict) -> "Outcome":
        actions: list[Action] = []
        for act_raw in dct["relevant_actions"]:
            actions.append(Action(**act_raw))
        load = dict(dct)
        load["relevant_actions"] = tuple(actions)
        return Outcome(**load)

    def __insert(self, before: bool, push: int, sac: int, action: Action) -> "Outcome":
        assert before or self.config == 0, f"{self.config=}"
        return Outcome(
            config=push + 2 * self.config if before else 0,
            full_config=push + 2 * self.full_config,
            end_slot=self.end_slot,
            sacrifice=sac + self.sacrifice,
            known=self.known,
            regret=self.regret,
            relevant_actions=tuple([action, *self.relevant_actions]),
        )

    def insert_canonical(self, before: bool, action: Action) -> "Outcome":
        return self.__insert(
            before=before,
            push=1,
            sac=0,
            action=action,
        )

    def insert_noncanonical(
        self, before: bool, is_adv_slot: bool, action: Action
    ) -> "Outcome":
        return self.__insert(
            before=before,
            push=0,
            sac=int(is_adv_slot),
            action=action,
        )


class Condition(ABC):
    @abstractmethod
    def known(self) -> bool:
        pass


@dataclass(frozen=True)
class UnknownCondition(Condition):
    def known(self) -> bool:
        return False


@dataclass(frozen=True)
class KnownCondition(Condition):
    adv_slots: int
    epoch_str: str

    def known(self) -> bool:
        return True


@dataclass(frozen=True)
class Reason:
    condition: Condition
    id: int


@dataclass(frozen=True)
class DetailedElement:
    value: np.float64
    probability: np.float64

    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class DetailedDistribution:
    slot: int

    distribution: tuple[DetailedElement, ...]
    id_to_outcome: MappingProxyType[int, Outcome]

    def __insert(
        self, old_id_to_new_outcome: dict[int, Outcome], sac: int
    ) -> "DetailedDistribution":
        return DetailedDistribution(
            slot=self.slot - 1,
            distribution=tuple(
                DetailedElement(
                    value=elem.value - sac,
                    probability=elem.probability,
                    reasons=tuple(
                        Reason(
                            condition=reason.condition,
                            id=old_id_to_new_outcome[reason.id].config,
                        )
                        for reason in elem.reasons
                    ),
                )
                for elem in self.distribution
            ),
            id_to_outcome=MappingProxyType(
                {outcome.config: outcome for outcome in old_id_to_new_outcome.values()}
            ),
        )

    def insert_canonical(
        self, is_adv_slot: bool, hide: bool = False
    ) -> "DetailedDistribution":
        old_id_to_new_config = {
            id: outcome.insert_canonical(
                before=self.slot <= 0,
                action=(
                    Action(slot=self.slot - 1, move=Move.HIDEBLOCK)
                    if hide
                    else (
                        Action(slot=self.slot - 1, move=Move.PROPOSEBLOCK)
                        if is_adv_slot
                        else Action(slot=self.slot - 1, move=Move.HONESTPROPOSE)
                    )
                ),
            )
            for id, outcome in self.id_to_outcome.items()
        }
        return self.__insert(old_id_to_new_config, sac=0)

    def insert_noncanonical(self, is_adv_slot: bool) -> "DetailedDistribution":
        old_id_to_new_config = {
            id: outcome.insert_noncanonical(
                before=self.slot < 0,
                is_adv_slot=is_adv_slot,
                action=(
                    Action(slot=self.slot - 1, move=Move.MISSBLOCK)
                    if is_adv_slot
                    else Action(slot=self.slot - 1, move=Move.FORKOUT)
                ),
            )
            for id, outcome in self.id_to_outcome.items()
        }
        return self.__insert(old_id_to_new_config, sac=int(is_adv_slot))

    def expected_value(self) -> np.float64:
        return sum(elem.value * elem.probability for elem in self.distribution)

    def expected_value_in_distribution(self) -> "DetailedDistribution":
        return DetailedDistribution(
            slot=self.slot,
            distribution=(
                DetailedElement(
                    value=self.expected_value(),
                    probability=np.float64(1.0),
                    reasons=tuple(
                        Reason(
                            condition=UnknownCondition(),
                            id=id,
                        )
                        for id in set(
                            reason.id
                            for elem in self.distribution
                            for reason in elem.reasons
                        )
                    ),
                ),
            ),
            id_to_outcome=MappingProxyType(
                {
                    id: Outcome(
                        config=outcome.config,
                        full_config=outcome.full_config,
                        end_slot=outcome.end_slot,
                        sacrifice=outcome.sacrifice,
                        known=False,
                        regret=False,
                        relevant_actions=outcome.relevant_actions,
                    )
                    for id, outcome in self.id_to_outcome.items()
                }
            ),
        )

    def max(self, other: "DetailedDistribution") -> "DetailedDistribution":
        assert self.slot == other.slot, f"{self.slot=} {other.slot=}"

        assert (
            len(set(self.id_to_outcome).intersection(other.id_to_outcome)) == 0
        ), f"{self.id_to_outcome=} {other.id_to_outcome=}"

        first_index = second_index = 0

        P_first = np.float64(0.0)  # P(self < self.distribution[first_index])
        P_second = np.float64(0.0)  # P(other < self.distribution[first_index])
        new_distrution: list[DetailedElement] = []

        while first_index < len(self.distribution) or second_index < len(
            other.distribution
        ):
            if first_index == len(self.distribution):
                new_distrution.append(other.distribution[second_index])
                second_index += 1
            elif second_index == len(other.distribution):
                new_distrution.append(self.distribution[first_index])
                first_index += 1
            elif (
                self.distribution[first_index].value
                > other.distribution[second_index].value
            ):
                if first_index > 0:

                    new_distrution.append(
                        DetailedElement(
                            value=other.distribution[second_index].value,
                            probability=P_first
                            * other.distribution[second_index].probability,
                            reasons=other.distribution[second_index].reasons,
                        )
                    )

                P_second += other.distribution[second_index].probability
                second_index += 1
            elif (
                self.distribution[first_index].value
                < other.distribution[second_index].value
            ):
                if second_index > 0:
                    new_distrution.append(
                        DetailedElement(
                            value=self.distribution[first_index].value,
                            probability=P_second
                            * self.distribution[first_index].probability,
                            reasons=self.distribution[first_index].reasons,
                        )
                    )

                P_first += self.distribution[first_index].probability
                first_index += 1
            else:
                assert (
                    self.distribution[first_index].value
                    == other.distribution[second_index].value
                ), f"{self.distribution[first_index].value=} {other.distribution[second_index].value=}"
                new_distrution.append(
                    DetailedElement(
                        value=self.distribution[first_index].value,
                        probability=P_first
                        * other.distribution[second_index].probability
                        + P_second * self.distribution[first_index].probability
                        + self.distribution[first_index].probability
                        * other.distribution[second_index].probability,
                        reasons=tuple(
                            [
                                *self.distribution[first_index].reasons,
                                *other.distribution[second_index].reasons,
                            ]
                        ),
                    )
                )

                P_second += other.distribution[second_index].probability
                second_index += 1

                P_first += self.distribution[first_index].probability
                first_index += 1

        return DetailedDistribution(
            slot=self.slot,
            distribution=tuple(new_distrution),
            id_to_outcome=MappingProxyType(
                {**self.id_to_outcome, **other.id_to_outcome}
            ),
        )

    def max_unknown(self, unknown: "DetailedDistribution") -> "DetailedDistribution":
        """
        Let's say X ~ self any Y ~ unknown
        Z := max(X, Y)
        We sample from X (Y is still unknown), and we aim to answer EXP(Z=z|X=x).
        This will be the result.

        Args:
            unknown (DetailedDistribution): Unknown distribution

        Returns:
            DetailedDistribution: z ~ Result <==> EXP(Z=z|X=x) x ~ self
        """

        assert (
            len(set(self.id_to_outcome).intersection(unknown.id_to_outcome)) == 0
        ), f"{self.id_to_outcome=} {unknown.id_to_outcome=}"

        unknown_exp_values = np.array(
            [elem.probability * elem.value for elem in unknown.distribution]
        )
        unknown_exp_values = np.cumsum(unknown_exp_values[::-1])[::-1]
        unknown_exp_values = np.append(unknown_exp_values, 0)

        probability_of_unknown_lesseq_than_x = np.float64(0.0)
        new_distribution: list[DetailedElement] = []

        second_index = 0
        for elem in self.distribution:
            while (
                second_index < len(unknown.distribution)
                and unknown.distribution[second_index].value <= elem.value
            ):
                probability_of_unknown_lesseq_than_x += unknown.distribution[
                    second_index
                ].probability
                second_index += 1

            new_distribution.append(
                DetailedElement(
                    value=probability_of_unknown_lesseq_than_x * elem.value
                    + unknown_exp_values[second_index],
                    probability=elem.probability,
                    reasons=elem.reasons,
                )
            )

        return DetailedDistribution(
            slot=self.slot,
            distribution=new_distribution,
            id_to_outcome=self.id_to_outcome,
        )

    def reset(self, regret: bool) -> "DetailedDistribution":
        return DetailedDistribution(
            slot=self.slot,
            distribution=self.distribution,
            id_to_outcome=MappingProxyType(
                {
                    id: Outcome(
                        config=outcome.config,
                        full_config=outcome.full_config,
                        end_slot=self.slot,
                        sacrifice=outcome.sacrifice,
                        known=outcome.known,
                        regret=regret,
                        relevant_actions=outcome.relevant_actions,
                    )
                    for id, outcome in self.id_to_outcome.items()
                }
            ),
        )
