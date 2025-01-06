from dataclasses import dataclass

from base.helpers import SLOTS_PER_EPOCH
from theory.method.utils.attack_strings import (
    EpochPostfix,
    EpochString,
    ExtendedAttackString,
)
from theory.method.quant.base import ForkQuantizedEAS, QuantizedEAS, SMQuantizedEAS
from theory.method.detailed_distribution import DetailedDistribution, Outcome


def quantize_group(
    distribution: DetailedDistribution,
    epoch_string_to_id: dict[str, int],
    num_of_epoch_strings: int,
) -> tuple[dict[int, Outcome], dict[int, list[int]]]:
    cfg_mapping = dict(distribution.id_to_outcome)

    result = {
        id: (
            [-1] * (num_of_epoch_strings * (SLOTS_PER_EPOCH + 1))
            if outcome.known
            else [-1]
        )
        for id, outcome in distribution.id_to_outcome.items()
    }

    for i, elem in enumerate(distribution.distribution):
        for reason in elem.reasons:
            if reason.condition.known():
                index = (
                    reason.condition.adv_slots * num_of_epoch_strings
                    + epoch_string_to_id[reason.condition.epoch_str]
                )
            else:
                index = 0

            assert result[reason.id][index] == -1, f"{reason=}"
            result[reason.id][index] = i

    return cfg_mapping, result


@dataclass
class StringEncoder:
    size_prefix: int
    size_postfix: int

    def encode_postf(self, postf: str) -> int:
        postf = postf.rjust(self.size_postfix, "h")
        postf_bin = postf.replace("a", "1").replace("h", "0")
        return int(postf_bin, 2)

    def encode_epoch_string(self, epoch_string: str) -> int:
        pref, postf = epoch_string.split("#")
        pref_num = len(pref)
        return pref_num + (self.size_prefix + 1) * self.encode_postf(postf)

    def encode_eas(self, eas: str) -> int:
        postf, epoch_string = eas.split(".")
        return self.encode_postf(postf) + (
            2**self.size_postfix
        ) * self.encode_epoch_string(epoch_string)


def get_mapping_by_eas_postf(
    size_prefix: int, size_postfix: int, eas_mapping: str
) -> dict[str, tuple[int, dict[str, int]]]:
    """
    Mapping from the postfix of the current EAS in epoch E.
    If EAS_E="AH.A#AAA" => we need to match epoch strings (in epoch E+2) to "AAA"

    Args:
        size_prefix: int
        size_postfix: int
        eas_mapping (str): mapping of extended attack strings

    Returns:
        dict[str, tuple[int, dict[str, int]]]: mapping from EpochPostfix's encoded int
        to a tuple. The first element is the number of possible epoch strings (=:num). The second
        one is a mapping from epoch strings to index, where 0 <= mapping[es] < num
    """
    result: dict[str, tuple[int, dict[str, int]]] = {}
    for postf in EpochPostfix.possibilities(size=size_postfix):
        mapping: dict[str, str] = {
            str(epoch_string): eas_mapping[f"{postf}.{epoch_string}"].split(".")[1]
            for epoch_string in EpochString.possibilities(
                size_pre=size_prefix, size_post=size_postfix
            )
        }
        values = list(set(mapping.values()))
        value_to_index = {value: i for i, value in enumerate(values)}
        result[postf.postfix] = (
            len(values),
            {es_from: value_to_index[es_to] for es_from, es_to in mapping.items()},
        )
    return result


class Quantizer:
    def __init__(
        self,
        size_prefix: int,
        size_postfix: int,
        eas_mapping: dict[str, str],
    ) -> QuantizedEAS:
        self.eas_mapping = eas_mapping
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix
        self.mapping_by_eas_postf = get_mapping_by_eas_postf(
            size_prefix=self.size_prefix,
            size_postfix=self.size_postfix,
            eas_mapping=self.eas_mapping,
        )
        self.eas_to_quant: dict[str, QuantizedEAS] = {}

    def quantize_sm(
        self, eas: ExtendedAttackString, distribution: DetailedDistribution
    ):
        num, mapping = self.mapping_by_eas_postf[eas.postfix_epoch_next.postfix]
        cfg_mapping, quantized_gr = quantize_group(
            distribution=distribution,
            epoch_string_to_id=mapping,
            num_of_epoch_strings=num,
        )

        num_of_epoch_strings, epoch_string_to_index = self.mapping_by_eas_postf[
            eas.postfix_epoch_next.postfix
        ]
        quant_model = SMQuantizedEAS(
            slot=distribution.slot,
            id_to_outcome=cfg_mapping,
            num_of_epoch_strings=num_of_epoch_strings,
            epoch_string_to_index=epoch_string_to_index,
            group=quantized_gr,
        )

        self.eas_to_quant[str(eas)] = quant_model

    def quantize_fork(
        self,
        eas: ExtendedAttackString,
        distribution: DetailedDistribution,
        regret_distributions: list[DetailedDistribution],
    ):
        num, mapping = self.mapping_by_eas_postf[eas.postfix_epoch_next.postfix]
        main_mapping, quantized_main_gr = quantize_group(
            distribution=distribution,
            epoch_string_to_id=mapping,
            num_of_epoch_strings=num,
        )

        regret = [
            quantize_group(
                distribution=distribution,
                epoch_string_to_id=mapping,
                num_of_epoch_strings=num,
            )
            for distribution in regret_distributions
        ]

        cfg_mapping = {}
        for mapping in [main_mapping, *[r[0] for r in regret]]:
            for key, val in mapping.items():
                if key in cfg_mapping:
                    assert cfg_mapping[key] == val
                cfg_mapping[key] = val

        num_of_epoch_strings, epoch_string_to_index = self.mapping_by_eas_postf[
            eas.postfix_epoch_next.postfix
        ]
        quant_model = ForkQuantizedEAS(
            slot=distribution.slot,
            id_to_outcome=cfg_mapping,
            num_of_epoch_strings=num_of_epoch_strings,
            epoch_string_to_index=epoch_string_to_index,
            head_group=quantized_main_gr,
            regret_groups=[group[1] for group in regret],
        )

        self.eas_to_quant[str(eas)] = quant_model
