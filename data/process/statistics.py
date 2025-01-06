from dataclasses import dataclass
import os
import pickle
from typing import Iterable, Optional

import yaml
from theory.method.preeval import PatternsWrapper, preeval_grinding
from base.beaconchain import BlockData, EpochAlternatives
from base.helpers import SLOTS_PER_EPOCH, Comparison, Status, git_status
from data.file_manager import FileManager
from data.process.data_to_string import DataToAttackString
from data.process.query import CaseOccurrences, query


@dataclass
class SlotExtracted:
    slot: int
    entity: str
    status: Status
    parent_slot: Optional[int]


@dataclass
class Case:
    attack_string: str
    block_statuses: str
    epoch: int
    attacker_candidate: str
    alpha: float
    extracted_slots: list[SlotExtracted]
    reality_to_entities: dict[str, Optional[list[str]]]

    def __str__(self) -> str:
        return self.name


@dataclass
class ComparedEpoch:
    comparison: Comparison
    reality_to_gain: dict[str, Optional[int]]


def is_matching(smaller: list, bigger: list) -> bool:
    return bigger[-len(smaller) :] == smaller


def most_similar(candidates: list[Iterable], gt: list) -> Iterable:
    return max(
        candidates, key=lambda candidate: sum(x == y for x, y in zip(candidate, gt))
    )


def evaluate_alternative_cases(
    alternatives: dict[int, EpochAlternatives],
    cases: dict[str, list[CaseOccurrences]],
    index_to_entity: dict[int, str],
    beaconchain: dict[int, BlockData],
) -> list[Case]:
    """

    Args:
        alternatives (dict[int, EpochAlternatives]): mapping from bool tensor of proposed/missed slots to alternative outcome
        cases (dict[str, list[CaseOccurrences]]): Concrete candidate epochs in reality when the attack might occured.

    Returns:
        list[Case]: list of Case's in reality, from this you can theorize whether there was an attack and draw correlations
        between different entities. One epoch can be part of more than one case (positive cases)
    """
    all_cases: list[Case] = []

    status_to_str: dict[int, str] = {1: "p", 2: "m", 3: "r"}

    for attack, case in cases.items():
        if attack == "." or attack.replace(".", "")[-1] != "a":
            continue
        possible_outcomes: list[str] = []
        possible_proposed_or_not: list[list[bool]] = []
        for occurrence in case:
            possible_outcomes.append(
                "".join(
                    status_to_str[block.status.value] for block in occurrence.pattern
                )
            )
            possible_proposed_or_not.append(
                [
                    block.status == Status.PROPOSED
                    for block in occurrence.pattern
                    if block.slot < 0
                ]
            )
        for i, occurrence in enumerate(case):

            for epoch in occurrence.occurrences:
                if epoch.epoch not in alternatives:
                    continue
                reality_mapping = alternatives[epoch.epoch].alternatives
                next_epoch_boundary = (epoch.epoch + 1) * SLOTS_PER_EPOCH
                head_index = occurrence.pattern[-1].slot

                try:
                    assert (
                        index_to_entity[
                            beaconchain[head_index + next_epoch_boundary].proposer_index
                        ]
                        == epoch.candidate
                    )
                    epoch_proposed_or_not = [
                        beaconchain[slot].status == Status.PROPOSED
                        for slot in range(
                            next_epoch_boundary - SLOTS_PER_EPOCH, next_epoch_boundary
                        )
                    ]
                    case = Case(
                        attack_string=attack,
                        block_statuses=possible_outcomes[i],
                        epoch=epoch.epoch,
                        attacker_candidate=epoch.candidate,
                        alpha=alternatives[epoch.epoch].entity_to_alpha[
                            epoch.candidate
                        ],
                        extracted_slots=[
                            SlotExtracted(
                                slot=block.slot + next_epoch_boundary,
                                entity=index_to_entity[
                                    beaconchain[
                                        block.slot + next_epoch_boundary
                                    ].proposer_index
                                ],
                                status=beaconchain[
                                    block.slot + next_epoch_boundary
                                ].status,
                                parent_slot=beaconchain[
                                    block.slot + next_epoch_boundary
                                ].parent_block_slot,
                            )
                            for block in occurrence.pattern
                        ],
                        reality_to_entities={},
                    )
                    for outcome, proposed_or_not in zip(
                        possible_outcomes, possible_proposed_or_not
                    ):
                        matching_keys = [
                            key
                            for key in reality_mapping
                            if is_matching(smaller=proposed_or_not, bigger=list(key))
                        ]
                        if len(matching_keys) == 0:
                            case.reality_to_entities[outcome] = None
                            continue
                        key = most_similar(matching_keys, epoch_proposed_or_not)
                        reality = [
                            index_to_entity[index] for index in reality_mapping[key]
                        ]
                        case.reality_to_entities[outcome] = reality
                    if len(case.reality_to_entities) > 0:
                        all_cases.append(case)
                except KeyError:
                    pass
    return all_cases


def eval_case(case: Case) -> ComparedEpoch:
    """

    Args:
        case (Case): -

    Returns:
        tuple[Comparison, dict[str, Optional[int]]]: Whether the candidate was better with the current scenario or could have done better.
        The second value in the tuple is the mapping from reality to gain
    """
    reality_to_gain: dict[str, Optional[int]] = {}

    for reality, entities in case.reality_to_entities.items():
        if entities is None:
            reality_to_gain[reality] = None
            continue
        reality_to_gain[reality] = entities.count(case.attacker_candidate)
        assert len(reality) == len(case.extracted_slots)
        reality_to_gain[reality] -= sum(
            [
                ext_slot.entity == case.attacker_candidate and stat != "p"
                for ext_slot, stat in zip(case.extracted_slots, reality)
            ]
        )

    comp: Comparison
    if any(gain is None for gain in reality_to_gain.values()):
        if any(
            reality_to_gain[case.block_statuses] < gain
            for gain in reality_to_gain.values()
            if gain is not None
        ):
            comp = Comparison.UNKNOWN_WORSE
        else:
            comp = Comparison.UNKNOWN

    else:
        other_gains = [
            gain
            for reality, gain in reality_to_gain.items()
            if reality != case.block_statuses
            if gain is not None
        ]
        if all(gain < reality_to_gain[case.block_statuses] for gain in other_gains):
            comp = Comparison.BEST
        elif all(gain <= reality_to_gain[case.block_statuses] for gain in other_gains):
            comp = Comparison.NEUTRAL
        else:
            comp = Comparison.WORSE

    return ComparedEpoch(
        comparison=comp,
        reality_to_gain=reality_to_gain,
    )


def export_statistics_to_cvs(
    cases: list[Case],
    folder: str,
    minimal_occurrence: int,
    only_summarized_stats: bool = False,
) -> None:

    str_to_status: dict[str, str] = {"p": "Proposed", "m": "Missed", "r": "Reorged"}
    owner_mapping: dict[str, str] = {"a": "Adversarial", "h": "Honest"}

    case_to_comparison: list[tuple[Case, ComparedEpoch]] = [
        (case, eval_case(case)) for case in cases
    ]
    ordered_cases = sorted(case_to_comparison, key=lambda x: x[1].comparison.value)
    lines: list[str] = []
    lines.append("Outcome;Attacker candidate;Stakes;Epoch;Attack string;Statuses")

    # Summarized table 1
    def to_str(num: Optional[int]) -> str:
        return str(num) if num is not None else "-"

    for case, compared_epoch in ordered_cases:
        realities = [
            case.block_statuses[1:],
            to_str(compared_epoch.reality_to_gain[case.block_statuses]),
        ]
        assert realities[-1] is not None
        for reality, gain in compared_epoch.reality_to_gain.items():
            assert reality[0] == "p", f"{reality=}"
            if reality != case.block_statuses:
                realities.extend([reality[1:], to_str(gain)])

        lines.append(
            f"{compared_epoch.comparison.name};{case.attacker_candidate};{case.alpha};{case.epoch};{case.attack_string.upper()};"
            + ";".join(realities)
            + ";"
        )

    cases_non_detailed_path = os.path.join(folder, "cases.csv")
    with open(cases_non_detailed_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    lines = []

    # Summarized table summed by outcome
    fields: list[str] = ["Attacker candidate"]
    for comp in Comparison:
        fields.append(comp.name)

    lines.append(";".join(fields) + ";")
    outcome_stats: dict[tuple[str, int], int] = {}
    overall_stats: dict[str, int] = (
        {}
    )  # Mapping from entity to its frequency in the cases
    for case, compared_epoch in ordered_cases:
        overall_stats[case.attacker_candidate] = (
            overall_stats.get(case.attacker_candidate, 0) + 1
        )
        outcome_stats[(case.attacker_candidate, compared_epoch.comparison.value)] = (
            outcome_stats.get(
                (case.attacker_candidate, compared_epoch.comparison.value), 0
            )
            + 1
        )
    entity_order_by_freq = sorted(
        overall_stats.items(), key=lambda x: x[1], reverse=True
    )
    for entity, freq in entity_order_by_freq:
        if freq < minimal_occurrence:
            break
        fields: list[str] = [entity]
        for comp in Comparison:
            fields.append(str(outcome_stats.get((entity, comp.value), 0)))
        lines.append(";".join(fields) + ";")

    summarized_path = os.path.join(folder, "summarized.csv")
    with open(summarized_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if only_summarized_stats:
        return

    lines = ["Outcome;Attacker candidate;Stakes;Epoch;Attack string;Statuses"]

    # Exporting all cases
    for case, compared_epoch in ordered_cases:
        lines.append(
            f"{compared_epoch.comparison.name};{case.attacker_candidate};{case.alpha};{case.epoch};{case.attack_string.upper()};{case.block_statuses[1:]}"
        )

        alternative_reality_keys = [
            key for key in case.reality_to_entities if key != case.block_statuses
        ]
        reality_keys = [case.block_statuses, *alternative_reality_keys]

        lines.append(
            "Slot;Entity;Parent Slot;Owner (A/H);"
            + ";".join([r[1:] for r in reality_keys])
            + ";"
        )
        attack_no_boundary = case.attack_string.replace(".", "")
        assert len(attack_no_boundary) == len(case.extracted_slots[1:]) and len(
            case.extracted_slots
        ) == len(case.block_statuses)

        for i, (attack_block, ext_slot) in enumerate(
            zip(attack_no_boundary, case.extracted_slots[1:])
        ):
            fields: list[str] = [
                str(ext_slot.slot),
                ext_slot.entity,
                to_str(ext_slot.parent_slot),
                owner_mapping[attack_block],
            ]
            for reality in reality_keys:

                fields.append(str_to_status[reality[i]])
            lines.append(";".join(fields) + ";")
        lines.append(f"Different outcomes in epoch {case.epoch+2};")

        lines.append(
            f";" + ";".join([r[1:] for r in reality_keys]) + ";"
        )  # This may include the first slot
        lines.append(
            "RANDAO outcome;"
            + ";".join(
                [
                    to_str(compared_epoch.reality_to_gain[reality])
                    for reality in reality_keys
                ]
            )
        )
        for slot_low in range(SLOTS_PER_EPOCH):

            fields: list[str] = [str(slot_low + (case.epoch + 2) * SLOTS_PER_EPOCH)]
            for reality in reality_keys:
                entities = case.reality_to_entities[reality]
                if entities is None:
                    fields.append("-")
                else:
                    fields.append(entities[slot_low])
            lines.append(";".join(fields) + ";")

        lines.append("")
        lines.append("")

    cases_detailed_path = os.path.join(folder, "cases_detailed.csv")
    with open(cases_detailed_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_statistics(
    file_manager: FileManager,
    size_prefix: int,
    size_postfix: int,
    export_folder: str,
    minimal_occurrence: int,
    only_summarized_stats: bool,
) -> None:
    preeval_mapping = preeval_grinding(
        size_prefix=size_prefix, size_postfix=size_postfix
    )
    assert preeval_mapping is not None
    patterns_wrapper = PatternsWrapper(mappings=preeval_mapping)
    data_to_attack_string = DataToAttackString(
        file_manager=file_manager, size_prefix=size_prefix, size_postfix=size_postfix
    )
    query_result = data_to_attack_string.query()

    alternatives: dict[int, EpochAlternatives]
    with open("data/processed_data/alternatives.pkl", "rb") as f:
        alternatives = pickle.load(f)

    cases = query(
        file_manager=file_manager,
        patterns_wrapper=patterns_wrapper,
        attack_string_query=query_result,
        alternatives=alternatives,
    )

    index_to_entity = file_manager.index_to_entity()
    res_cases = evaluate_alternative_cases(
        alternatives=alternatives,
        cases=cases,
        index_to_entity=index_to_entity,
        beaconchain=file_manager.beaconchain(),
    )

    os.makedirs(export_folder, exist_ok=True)
    git_state, diff = git_status()
    if diff:
        diff_path = os.path.join(export_folder, "diff.patch")
        with open(diff_path, "w", encoding="utf-8") as diff_file:
            diff_file.write(diff)

    config = git_state

    config.update(
        {
            "epochs": {
                "min": min(alternatives),
                "max": max(alternatives),
            },
            "size_prefix": size_prefix,
            "size_postfix": size_postfix,
            "minimal_occurrence": minimal_occurrence,
            "only_summarized_stats": only_summarized_stats,
        }
    )
    config_path = os.path.join(export_folder, "config.yaml")
    with open(config_path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    export_statistics_to_cvs(
        cases=res_cases,
        folder=export_folder,
        minimal_occurrence=minimal_occurrence,
        only_summarized_stats=only_summarized_stats,
    )
