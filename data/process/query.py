from dataclasses import dataclass, field

from theory.method.preeval import PatternsWrapper
from base.beaconchain import BlockData, EpochAlternatives
from base.shuffle import SLOTS_PER_EPOCH
from data.file_manager import FileManager
from data.process.data_to_string import EpochAttackStringAndCandidate


UPPER_SELFISH_MIXING = 8


@dataclass
class EpochInfo:
    epoch: int
    candidate: str


@dataclass
class CaseOccurrences:
    pattern: list[BlockData]
    occurrences: list[EpochInfo] = field(default_factory=list)


def query(
    file_manager: FileManager,
    patterns_wrapper: PatternsWrapper,
    attack_string_query: dict[int, list[EpochAttackStringAndCandidate]],
    alternatives: dict[int, EpochAlternatives],
) -> dict[str, list[CaseOccurrences]]:
    beaconchain = file_manager.beaconchain()

    result: dict[str, list[CaseOccurrences]] = {}
    for epoch, attack_string_candidates in attack_string_query.items():
        epoch_boundary = (epoch + 1) * SLOTS_PER_EPOCH

        for as_candidate in attack_string_candidates:
            if epoch not in alternatives:
                continue

            alpha = alternatives[epoch].entity_to_alpha[as_candidate.candidate]
            attack_string_mapping, patterns = patterns_wrapper[alpha]
            act_attack_string = attack_string_mapping[as_candidate.attack_string]
            if act_attack_string == ".":
                continue
            act_possib_patterns = patterns[act_attack_string]
            if act_attack_string not in result:
                result[act_attack_string] = [
                    CaseOccurrences(pattern=pattern, occurrences=[])
                    for pattern in act_possib_patterns
                ]

            for case_occurrence in result[act_attack_string]:
                matching = True
                for pattern_block in case_occurrence.pattern:
                    if not beaconchain[epoch_boundary + pattern_block.slot].is_matching(
                        pattern_block
                    ):
                        matching = False
                        break

                if matching:
                    case_occurrence.occurrences.append(
                        EpochInfo(epoch=epoch, candidate=as_candidate.candidate)
                    )

    return result
