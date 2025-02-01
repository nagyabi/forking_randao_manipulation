from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar

import numpy as np

from theory.method.utils.attack_strings import (
    EpochPostfix,
    EpochString,
    ExtendedAttackString,
)
from base.helpers import SLOTS_PER_EPOCH
from scipy.stats import binom

from theory.method.detailed_distribution import (
    DetailedDistribution,
    DetailedElement,
    KnownCondition,
    Outcome,
    Reason,
    T_Detailed_Distribution,
)


class DistributionLike(Protocol):
    def expected_value_in_distribution(self: "T_Distribution") -> "T_Distribution":
        pass

    def increase_sacrifice(self: "T_Distribution", sacrifice: int) -> "T_Distribution":
        pass

    def max(self: "T_Distribution", other: "T_Distribution") -> "T_Distribution":
        pass

    def max_unknown(
        self: "T_Distribution", unknown: "T_Distribution"
    ) -> "T_Distribution":
        pass


T_Distribution = TypeVar("T_Distribution", bound=DistributionLike)


@dataclass
class Element:
    value: np.float64
    probability: np.float64


@dataclass
class RichElement:
    value: np.float64
    probability: np.float64

    expected_value_of_vars: np.ndarray
    # expected_value_of_vars[0] -> RO
    # expected_value_of_vars[1] -> adv slots
    # expected_value_of_vars[2] -> forked honest slots
    # expected_value_of_vars[3] -> probability of forking
    # expected_value_of_vars[4] -> probability of regret but no fork
    # expected_value_of_vars[5] -> probability of missing for sm, no case of the above
    # expected_value_of_vars[6] -> expected value when forking
    # expected_value_of_vars[7] -> expected value when regretting but not forking
    # expected_value_of_vars[8] -> expected value when missing for sm, no case of above

    distribution_of_epoch_strings: np.ndarray

    # invariant: probability == np.sum(distribution_of_epoch_string)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, RichElement):
            return all(
                [
                    self.value == value.value,
                    self.probability == value.probability,
                    np.all(self.expected_value_of_vars == value.expected_value_of_vars),
                    np.all(
                        self.distribution_of_epoch_strings
                        == value.distribution_of_epoch_strings
                    ),
                ]
            )
        return False


@dataclass
class ValuedDistribution:
    """
    Distribution of the value function + RO. For optimization only the value function is present in each
    Invariant: distribution is sorted by elem.value (for all elem: Element in self.distribution)
    """

    distribution: list[Element]

    def expected_value(self) -> np.float64:
        return sum(elem.probability * elem.value for elem in self.distribution)

    def expected_value_in_distribution(self) -> "ValuedDistribution":
        return ValuedDistribution(
            distribution=[
                Element(
                    value=self.expected_value(),
                    probability=np.float64(1.0),
                )
            ]
        )

    def increase_sacrifice(self, sacrifice: int) -> "ValuedDistribution":
        new_distribution = [
            Element(value=elem.value - sacrifice, probability=elem.probability)
            for elem in self.distribution
        ]
        return ValuedDistribution(distribution=new_distribution)

    def max(self, other: "ValuedDistribution") -> "ValuedDistribution":
        """

        Args:
            other (ValuedDistribution): Other distribution

        Returns:
            ValuedDistribution: The maximum of the two probability variables: 'self' and 'other'
        """
        first_index = second_index = 0  # First refers to self, second to other
        P_first = np.float64(0.0)  # P(self < self.distribution[first_index])
        P_second = np.float64(0.0)  # P(other < self.distribution[first_index])
        new_distrution: list[Element] = []

        while first_index < len(self.distribution) or second_index < len(
            other.distribution
        ):
            if first_index == len(self.distribution):
                new_distrution.append(deepcopy(other.distribution[second_index]))
                second_index += 1
            elif second_index == len(other.distribution):
                new_distrution.append(deepcopy(self.distribution[first_index]))
                first_index += 1
            elif (
                self.distribution[first_index].value
                > other.distribution[second_index].value
            ):
                if first_index > 0:

                    new_distrution.append(
                        Element(
                            value=other.distribution[second_index].value,
                            probability=P_first
                            * other.distribution[second_index].probability,
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
                        Element(
                            value=self.distribution[first_index].value,
                            probability=P_second
                            * self.distribution[first_index].probability,
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
                    Element(
                        value=self.distribution[first_index].value,
                        probability=P_first
                        * other.distribution[second_index].probability
                        + P_second * self.distribution[first_index].probability
                        + self.distribution[first_index].probability
                        * other.distribution[second_index].probability,
                    )
                )

                P_second += other.distribution[second_index].probability
                second_index += 1

                P_first += self.distribution[first_index].probability
                first_index += 1

        return ValuedDistribution(distribution=new_distrution)

    def max_unknown(self, unknown: "ValuedDistribution") -> "ValuedDistribution":
        """
        Let's say X ~ self any Y ~ unknown
        Z := max(X, Y)
        We sample from X (Y is still unknown), and we aim to answer EXP(Z=z|X=x).
        This will be the result.

        Args:
            unknown (ValuedDistribution): Unknown distribution

        Returns:
            ValuedDistribution: z ~ Result <==> EXP(Z=z|X=x) x ~ self
        """

        unknown_exp_values = np.array(
            [elem.probability * elem.value for elem in unknown.distribution]
        )
        unknown_exp_values = np.cumsum(unknown_exp_values[::-1])[::-1]
        unknown_exp_values = np.append(unknown_exp_values, 0)

        probability_of_unknown_lesseq_than_x = np.float64(0.0)
        new_distribution: list[Element] = []

        second_index = 0
        for x in self.distribution:
            while (
                second_index < len(unknown.distribution)
                and unknown.distribution[second_index].value <= x.value
            ):
                probability_of_unknown_lesseq_than_x += unknown.distribution[
                    second_index
                ].probability
                second_index += 1

            new_distribution.append(
                Element(
                    value=probability_of_unknown_lesseq_than_x * x.value
                    + unknown_exp_values[second_index],
                    probability=x.probability,
                )
            )

        return ValuedDistribution(distribution=new_distribution)


@dataclass(frozen=True)
class ApproximatedDistribution:
    distribution: np.ndarray

    def expected_value(self) -> np.float64:
        return np.average(self.distribution)
    
    def expected_value_in_distribution(self) -> "ApproximatedDistribution":
        exp_val = self.expected_value()
        return ApproximatedDistribution(np.array([exp_val for _ in self.distribution], dtype=np.float64))
    
    def increase_sacrifice(self, sacrifice: int) -> "ApproximatedDistribution":
        return ApproximatedDistribution(distribution=self.distribution - sacrifice)
    
    def max(self, other: "ApproximatedDistribution") -> "ApproximatedDistribution":
        array1 = np.repeat(self.distribution[:, np.newaxis], len(other.distribution), axis=1)
        array2 = np.repeat(other.distribution[np.newaxis, :], len(self.distribution), axis=0)
        try:
            max_array = np.maximum(array1, array2)
        except Exception as e:

            print(f"{array1.shape=} {array2.shape=}")
            print(f"{array1=}")
            print(f"{array2=}")
            print()
            raise e
        sorted_array = np.sort(max_array.flatten())
        array = sorted_array.reshape((len(other.distribution), len(self.distribution)))
        return ApproximatedDistribution(np.average(array, axis=0))
    
    def max_unknown(self, unknown: "ApproximatedDistribution") -> "ApproximatedDistribution":
        array1 = np.repeat(self.distribution[:, np.newaxis], len(unknown.distribution), axis=1)
        array2 = np.repeat(unknown.distribution[np.newaxis, :], len(self.distribution), axis=0)
        max_array = np.maximum(array1, array2)
        array = np.average(max_array, axis=1)
        return ApproximatedDistribution(distribution=array)

@dataclass
class RichValuedDistribution:
    distribution: list[RichElement]

    def expected_value(self) -> np.float64:
        return sum(elem.probability * elem.value for elem in self.distribution)

    def expected_value_in_distribution(self) -> "RichValuedDistribution":
        return RichValuedDistribution(
            distribution=[
                RichElement(
                    value=self.expected_value(),
                    probability=np.float64(1.0),
                    expected_value_of_vars=sum(
                        elem.expected_value_of_vars for elem in self.distribution
                    ),
                    distribution_of_epoch_strings=sum(
                        elem.distribution_of_epoch_strings for elem in self.distribution
                    ),
                )
            ]
        )

    def increase_sacrifice(self, sacrifice: int) -> "RichValuedDistribution":
        new_distribution = [
            RichElement(
                value=elem.value - sacrifice,
                probability=elem.probability,
                expected_value_of_vars=np.array(
                    [
                        elem.expected_value_of_vars[0] - sacrifice * elem.probability,
                        *elem.expected_value_of_vars[1:5],
                        (
                            elem.probability
                            - elem.expected_value_of_vars[3]
                            - elem.expected_value_of_vars[4]
                            if sacrifice > 0
                            else elem.expected_value_of_vars[5]
                        ),
                        *(
                            elem.expected_value_of_vars[6:8]
                            - elem.expected_value_of_vars[3:5] * sacrifice
                        ),
                        (
                            elem.expected_value_of_vars[0]
                            - np.sum(elem.expected_value_of_vars[6:8])
                            - sacrifice
                            * (
                                elem.probability
                                - np.sum(elem.expected_value_of_vars[3:5])
                            )
                            if sacrifice > 0
                            else elem.expected_value_of_vars[8]
                            - elem.expected_value_of_vars[5] * sacrifice
                        ),
                    ],
                    dtype=np.float64,
                ),
                distribution_of_epoch_strings=elem.distribution_of_epoch_strings,
            )
            for elem in self.distribution
        ]
        return RichValuedDistribution(distribution=new_distribution)

    def regret(self) -> "RichValuedDistribution":
        new_distribution = [
            RichElement(
                value=elem.value,
                probability=elem.probability,
                expected_value_of_vars=np.array(
                    [
                        *elem.expected_value_of_vars[:4],
                        elem.probability - elem.expected_value_of_vars[3],
                        0.0,
                        elem.expected_value_of_vars[6],
                        elem.expected_value_of_vars[0] - elem.expected_value_of_vars[6],
                        0.0,
                    ],
                    dtype=np.float64,
                ),
                distribution_of_epoch_strings=elem.distribution_of_epoch_strings,
            )
            for elem in self.distribution
        ]
        return RichValuedDistribution(distribution=new_distribution)

    def increase_forked_honest_blocks(
        self, forked_honest_blocks: int
    ) -> "RichValuedDistribution":
        new_distribution = [
            RichElement(
                value=elem.value,
                probability=elem.probability,
                expected_value_of_vars=np.array(
                    [
                        elem.expected_value_of_vars[0],
                        elem.expected_value_of_vars[1],
                        elem.expected_value_of_vars[2]
                        + forked_honest_blocks * elem.probability,
                        elem.probability,
                        0,
                        0,
                        elem.expected_value_of_vars[0],
                        0,
                        0,
                    ],
                    dtype=np.float64,
                ),
                distribution_of_epoch_strings=elem.distribution_of_epoch_strings,
            )
            for elem in self.distribution
        ]
        return RichValuedDistribution(distribution=new_distribution)

    def max(self, other: "RichValuedDistribution") -> "RichValuedDistribution":
        """

        Args:
            other (RichValuedDistribution): Other distribution

        Returns:
            RichValuedDistribution: The maximum of the two probability variables: 'self' and 'other'
        """
        first_index = second_index = 0  # First refers to self, second to other
        P_first = np.float64(0.0)  # P(self < self.distribution[first_index])
        P_second = np.float64(0.0)  # P(other < self.distribution[first_index])
        new_distrution: list[RichElement] = []

        while first_index < len(self.distribution) or second_index < len(
            other.distribution
        ):
            if first_index == len(self.distribution):
                new_distrution.append(deepcopy(other.distribution[second_index]))
                second_index += 1
            elif second_index == len(other.distribution):
                new_distrution.append(deepcopy(self.distribution[first_index]))
                first_index += 1
            elif (
                self.distribution[first_index].value
                > other.distribution[second_index].value
            ):
                if first_index > 0:

                    new_distrution.append(
                        RichElement(
                            value=other.distribution[second_index].value,
                            probability=P_first
                            * other.distribution[second_index].probability,
                            expected_value_of_vars=P_first
                            * other.distribution[second_index].expected_value_of_vars,
                            distribution_of_epoch_strings=P_first
                            * other.distribution[
                                second_index
                            ].distribution_of_epoch_strings,
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
                        RichElement(
                            value=self.distribution[first_index].value,
                            probability=P_second
                            * self.distribution[first_index].probability,
                            expected_value_of_vars=P_second
                            * self.distribution[first_index].expected_value_of_vars,
                            distribution_of_epoch_strings=P_second
                            * self.distribution[
                                first_index
                            ].distribution_of_epoch_strings,
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
                    RichElement(
                        value=self.distribution[first_index].value,
                        probability=P_first
                        * other.distribution[second_index].probability
                        + P_second * self.distribution[first_index].probability
                        + self.distribution[first_index].probability
                        * other.distribution[second_index].probability,
                        expected_value_of_vars=P_first
                        * other.distribution[second_index].expected_value_of_vars
                        + (P_second + other.distribution[second_index].probability)
                        * self.distribution[first_index].expected_value_of_vars,
                        distribution_of_epoch_strings=P_first
                        * other.distribution[second_index].distribution_of_epoch_strings
                        + P_second
                        * self.distribution[first_index].distribution_of_epoch_strings
                        + self.distribution[first_index].distribution_of_epoch_strings
                        * other.distribution[second_index].probability,
                    )
                )

                P_second += other.distribution[second_index].probability
                second_index += 1

                P_first += self.distribution[first_index].probability
                first_index += 1

        return RichValuedDistribution(distribution=new_distrution)

    def max_unknown(
        self, unknown: "RichValuedDistribution"
    ) -> "RichValuedDistribution":
        """
        Let's say X ~ self any Y ~ unknown
        Z := max(X, Y)
        We sample from X (Y is still unknown), and we aim to answer EXP(Z=z|X=x).
        This will be the result.

        Args:
            unknown (RichValuedDistribution): Unknown distribution

        Returns:
            RichValuedDistribution: z ~ Result <==> EXP(Z=z|X=x) x ~ self
        """

        unknown_exp_values = np.array(
            [elem.probability * elem.value for elem in unknown.distribution],
            dtype=np.float64,
        )
        unknown_exp_values = np.cumsum(unknown_exp_values[::-1])[::-1]
        unknown_exp_values = np.append(unknown_exp_values, np.float64(0.0))

        unknown_vars = np.vstack(
            [elem.expected_value_of_vars for elem in reversed(unknown.distribution)]
        )
        unknown_vars = np.cumsum(unknown_vars, axis=0)[::-1]
        unknown_vars = np.vstack([unknown_vars, np.zeros(9, dtype=np.float64)])

        unknown_epoch_strs = np.vstack(
            [
                elem.distribution_of_epoch_strings
                for elem in reversed(unknown.distribution)
            ]
        )
        unknown_epoch_strs = np.cumsum(unknown_epoch_strs, axis=0)[::-1, :]
        unknown_epoch_strs = np.vstack(
            [
                unknown_epoch_strs,
                np.zeros(shape=(unknown_epoch_strs.shape[-1],), dtype=np.float64),
            ]
        )

        probability_of_unknown_lesseq_than_x = np.float64(0.0)

        new_distribution: list[RichElement] = []

        second_index = 0
        for x in self.distribution:
            while (
                second_index < len(unknown.distribution)
                and unknown.distribution[second_index].value <= x.value
            ):
                probability_of_unknown_lesseq_than_x += unknown.distribution[
                    second_index
                ].probability
                second_index += 1

            new_distribution.append(
                RichElement(
                    value=probability_of_unknown_lesseq_than_x * x.value
                    + unknown_exp_values[second_index],
                    probability=x.probability,
                    expected_value_of_vars=probability_of_unknown_lesseq_than_x
                    * x.expected_value_of_vars
                    + unknown_vars[second_index] * x.probability,
                    distribution_of_epoch_strings=probability_of_unknown_lesseq_than_x
                    * x.distribution_of_epoch_strings
                    + unknown_epoch_strs[second_index] * x.probability,
                )
            )

        return RichValuedDistribution(distribution=new_distribution)


class DistributionMaker(ABC, Generic[T_Detailed_Distribution]):
    @abstractmethod
    def make_detailed(
        self,
        value_function: dict[str, np.float64],
        postfix_next_epoch: EpochPostfix,
        nullcase: str,
    ) -> T_Detailed_Distribution:
        pass


class MakerBase:
    def __init__(
        self,
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
        eas_mapping: dict[str, str],
    ) -> None:
        self.alpha = alpha
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix
        self.eas_mapping = eas_mapping

        size = self.size_postfix + self.size_prefix

        self.epoch_string_to_chances: dict[str, list[RichElement]] = {}

        slots_in_string, all_slots = np.indices(
            (SLOTS_PER_EPOCH + 1, SLOTS_PER_EPOCH + 1)
        )
        chances = (
            binom.pmf(all_slots - slots_in_string, SLOTS_PER_EPOCH - size, alpha)
            * alpha**slots_in_string
            * (1 - alpha) ** (size - slots_in_string)
        )

        self.epoch_string_possibilities = EpochString.possibilities(
            size_pre=size_prefix, size_post=size_postfix
        )
        self.epoch_string_to_index: dict[str, int] = {
            str(epoch_string): index
            for index, epoch_string in enumerate(self.epoch_string_possibilities)
        }

        for epoch_string in self.epoch_string_possibilities:
            possibilities: list[RichElement] = []
            for all_slot in range(SLOTS_PER_EPOCH + 1):
                probability: np.float64 = np.float64(0.0)
                for meaning_pre in epoch_string.prefix.meanings:
                    for meaning_post in epoch_string.postfix.meanings:
                        slots_in_string = meaning_pre.count("a") + meaning_post.count(
                            "a"
                        )

                        probability += chances[slots_in_string, all_slot]
                dist_of_es = np.zeros(
                    len(self.epoch_string_possibilities), dtype=np.float64
                )
                dist_of_es[self.epoch_string_to_index[str(epoch_string)]] = probability
                possibilities.append(
                    RichElement(
                        value=all_slot,
                        probability=probability,
                        expected_value_of_vars=np.array(
                            [
                                probability * all_slot,
                                probability * all_slot,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ],
                            dtype=np.float64,
                        ),
                        distribution_of_epoch_strings=dist_of_es,
                    )
                )
            self.epoch_string_to_chances[str(epoch_string)] = possibilities


class DistMakerBase(ABC, Generic[T_Distribution]):
    @abstractmethod
    def make_distribution(
        self,
        **kwargs,
    ) -> T_Distribution:
        pass


class ValuedDistributionMaker(MakerBase, DistMakerBase[ValuedDistribution]):
    def make_distribution(
        self,
        value_function: dict[str, np.float64],
        postfix_next_epoch: EpochPostfix,
    ) -> ValuedDistribution:
        """
        Generates a distribution based on the end of epoch E+1 (postfix_next_epoch)
        Some strings might encode multiple possibilities, it considers them.

        Args:
            value_function (dict[str, np.float64]): mapping from extended_attack_string (in string repr) to the value.
            postfix_next_epoch (EpochPostfix): End of epoch E+1.

        Returns:
            ValuedDistribution: Distribution of values if we sample randomly an epoch (E+2)
        """

        distribution: list[Element] = []
        for epoch_string, elements in self.epoch_string_to_chances.items():
            next_eas_repr = self.eas_mapping[f"{postfix_next_epoch}.{epoch_string}"]
            for element in elements:
                distribution.append(
                    Element(
                        value=element.value + value_function[next_eas_repr],
                        probability=element.probability,
                    )
                )

        value_to_element: dict[np.float64, Element] = {}
        for element in distribution:
            if element.value not in value_to_element:
                value_to_element[element.value] = Element(
                    value=element.value, probability=np.float64(0.0)
                )
            value_to_element[element.value].probability += element.probability

        distribution = [value_to_element[key] for key in sorted(list(value_to_element))]

        return ValuedDistribution(distribution=distribution)

class ApproximatedDistributionMaker(MakerBase, DistMakerBase[ApproximatedDistribution]):
    def __init__(
        self,
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
        eas_mapping: dict[str, str],
        memory_size: int,
    ) -> None:
        super().__init__(alpha, size_prefix, size_postfix, eas_mapping)
        self.memory_size = memory_size
    
    def make_distribution(self, value_function: dict[str, np.float64], postfix_next_epoch: EpochPostfix) -> ApproximatedDistribution:
        value_to_probability: dict[float, float] = defaultdict(float)
        for epoch_string, elements in self.epoch_string_to_chances.items():
            next_eas_repr = self.eas_mapping[f"{postfix_next_epoch}.{epoch_string}"]
            for element in elements:
                value = element.value + value_function[next_eas_repr]
                value_to_probability[value] += element.probability
        
        sorted_value_to_probability = sorted(value_to_probability.items(), key=lambda x: x[0])
        result: list[float] = []
        cluster: list[float] = []
        acc_prob = 0
        chunk = 1 / self.memory_size
        for value, prob in sorted_value_to_probability:
            while acc_prob + prob >= chunk:
                assert acc_prob <= chunk, f"{acc_prob=} {chunk=}"
                rest = acc_prob + prob - chunk
                cluster.append(value * (prob - rest))
                result.append(sum(cluster) * self.memory_size)
                cluster = []
                acc_prob = 0
                prob = rest
            acc_prob += prob
            cluster.append(prob * value)

        assert abs(len(result) - self.memory_size) <= 1
        if len(result) < self.memory_size:
            assert len(cluster) > 0
            result.append(sum(cluster) * self.memory_size)
        assert len(result) == self.memory_size
        array = np.array(result, dtype=np.float64)
        assert np.isclose(np.average(array), sum(value * prob for value, prob in sorted_value_to_probability))
        return ApproximatedDistribution(array)


class RichDistributionMaker(MakerBase, DistMakerBase[RichValuedDistribution]):
    def __init__(
        self,
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
        eas_mapping: dict[str, str],
    ):
        super().__init__(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            eas_mapping=eas_mapping,
        )

        self.postfix_to_meanings = {
            postfix_next_epoch.postfix: list(
                set(
                    self.eas_mapping[f"{postfix_next_epoch}.{epoch_string}"]
                    for epoch_string in self.epoch_string_possibilities
                )
            )
            for postfix_next_epoch in EpochPostfix.possibilities(size=self.size_postfix)
        }

    def make_distribution(
        self,
        value_function: dict[str, np.float64],
        postfix_next_epoch: str | EpochPostfix,
    ) -> RichValuedDistribution:
        """
        Generates a distribution based on the end of epoch E+1 (postfix_next_epoch)
        Some strings might encode multiple possibilities, it considers them.

        Args:
            value_function (dict[str, np.float64]): mapping from extended_attack_string (in string repr) to the value.
            postfix_next_epoch (EpochPostfix): End of epoch E+1.

        Returns:
            RichValuedDistribution: Distribution of values if we sample randomly an epoch (E+2)
        """

        rich_distribution: list[RichElement] = []
        for epoch_string, elements in self.epoch_string_to_chances.items():
            next_eas_repr = self.eas_mapping[f"{postfix_next_epoch}.{epoch_string}"]
            for element in elements:
                rich_distribution.append(
                    RichElement(
                        value=element.value + value_function[next_eas_repr],
                        probability=element.probability,
                        expected_value_of_vars=element.expected_value_of_vars,
                        distribution_of_epoch_strings=np.copy(
                            element.distribution_of_epoch_strings
                        ),
                    )
                )

        value_to_r_element: dict[np.float64, RichElement] = {}
        for r_element in rich_distribution:
            if r_element.value not in value_to_r_element:
                value_to_r_element[r_element.value] = RichElement(
                    value=r_element.value,
                    probability=np.float64(0.0),
                    expected_value_of_vars=np.zeros(9, dtype=np.float64),
                    distribution_of_epoch_strings=np.zeros(
                        len(self.epoch_string_possibilities), dtype=np.float64
                    ),
                )
            value_to_r_element[r_element.value].probability += r_element.probability
            value_to_r_element[
                r_element.value
            ].expected_value_of_vars += r_element.expected_value_of_vars
            value_to_r_element[
                r_element.value
            ].distribution_of_epoch_strings += r_element.distribution_of_epoch_strings

        rich_distribution = [
            value_to_r_element[key] for key in sorted(list(value_to_r_element))
        ]

        meaning_to_index = {
            meaning: index
            for index, meaning in enumerate(
                self.postfix_to_meanings[str(postfix_next_epoch)]
            )
        }
        for r_elem in rich_distribution:
            new_eas_dist = np.zeros(
                len(self.postfix_to_meanings[str(postfix_next_epoch)]), dtype=np.float64
            )
            for i, prob in enumerate(r_elem.distribution_of_epoch_strings):
                next_eas = self.eas_mapping[
                    f"{postfix_next_epoch}.{self.epoch_string_possibilities[i]}"
                ]
                index = meaning_to_index[next_eas]
                new_eas_dist[index] += prob
            r_elem.distribution_of_epoch_strings = new_eas_dist

        return RichValuedDistribution(distribution=rich_distribution)


class DetailedDistributionMaker(MakerBase, DistributionMaker[DetailedDistribution]):
    def make_detailed(
        self,
        value_function: dict[str, np.float64],
        postfix_next_epoch: EpochPostfix,
        nullcase: str,
    ) -> DetailedDistribution:
        """
        Makes a detailed distribution from the nullcase.

        Args:
            value_function (dict[str, np.float64]): value function determining value: float64 for each EAS
            postfix_next_epoch (EpochPostfix): in the eas, the last part, e.g. "AH.A#AAH" => "AAH"
            nullcase (str): in {".", ".a", ".ha" ...}, any null case where the attack string's slots before the epoch boundary are all honest

        Returns:
            DetailedDistribution: corresponding distribution with the correct slot
        """

        value_to_elements: dict[np.float64, list[DetailedElement]] = {}
        for epoch_string, elements in self.epoch_string_to_chances.items():
            next_eas_repr = self.eas_mapping[f"{postfix_next_epoch}.{epoch_string}"]
            same = epoch_string == next_eas_repr.split(".")[1]
            for element in elements:
                value = element.value + value_function[next_eas_repr]
                if value not in value_to_elements:
                    value_to_elements[value] = []

                value_to_elements[value].append(
                    DetailedElement(
                        value=element.value + value_function[next_eas_repr],
                        probability=element.probability,
                        reasons=(
                            (
                                Reason(
                                    condition=KnownCondition(
                                        adv_slots=element.value,
                                        epoch_str=epoch_string,
                                    ),
                                    id=0,
                                ),
                            )
                            if same
                            else ()
                        ),
                    )
                )

        values: list[np.float64] = sorted(list(value_to_elements))
        distribution = tuple(
            [
                DetailedElement(
                    value=value,
                    probability=sum(
                        elem.probability for elem in value_to_elements[value]
                    ),
                    reasons=tuple(
                        reason
                        for elem in value_to_elements[value]
                        for reason in elem.reasons
                    ),
                )
                for value in values
            ]
        )

        slot = 0 if nullcase == "." else len(nullcase) - 2
        return DetailedDistribution(
            slot=slot,
            distribution=distribution,
            id_to_outcome=MappingProxyType(
                {
                    0: Outcome(
                        config=0,
                        full_config=int(nullcase != "."),
                        end_slot=slot,
                        sacrifice=0,
                        known=True,
                        regret=False,
                        det_slot_statuses=(),
                    )
                }
            ),
        )


class DistributionMaxer(Generic[T_Distribution]):
    def __init__(self, base_distribution: T_Distribution) -> None:
        self.__distributions = [base_distribution]

    def max(self, number_of_dist: int) -> T_Distribution:
        """
        Args:
            number_of_dist (int): number of base distributions

        Returns:
            T_Distribution: max distribution of number_of_dist base distributions
        """
        assert number_of_dist > 0, f"{number_of_dist=} should be positive"

        dists: list[T_Distribution] = []
        digit = 0
        while number_of_dist > 0:
            if digit >= len(self.__distributions):
                latest = self.__distributions[-1]
                self.__distributions.append(latest.max(latest))

            if number_of_dist % 2 == 1:
                dists.append(self.__distributions[digit])

            number_of_dist //= 2
            digit += 1

        res = dists[0]
        for dist in dists[1:]:
            res = res.max(dist)

        return res


def selfish_mixing(base_distribution: T_Distribution, slots: int) -> T_Distribution:
    """

    Args:
        base_distribution (T_Distribution): base distribution
        slots (int): number of slots before the distribution to choose to MISS/PROPOSE

    Returns:
        T_Distribution: distribution after missing/proposing before base_distribution
    """
    result = base_distribution
    for _ in range(slots):
        result = result.max(result.increase_sacrifice(sacrifice=1))

    return result


if __name__ == "__main__":
    size_pref = 2
    size_post = 7
    maker = ValuedDistributionMaker(
        alpha=np.float64(0.201), size_prefix=size_pref, size_postfix=size_post
    )
    value_function = {
        str(eas): np.float64(0.0)
        for eas in ExtendedAttackString.possibilities(
            size_prefix=size_pref, size_postfix=size_post
        )
    }
    print(f"{value_function=}")
    dist = maker.make_distribution(
        value_function=value_function, postfix_next_epoch="aa"
    )
    a = selfish_mixing(dist, 4)

    print(dist.max(dist.increase_sacrifice(1)).expected_value())
