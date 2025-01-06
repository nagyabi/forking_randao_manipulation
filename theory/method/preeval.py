from copy import deepcopy
import math
import os
from typing import Optional
import numpy as np
from tqdm import tqdm

from theory.method.utils.attack_strings import AttackString
from theory.method.utils.voting import dynamic_vote
from base.beaconchain import BlockData, PreEvaledTyp, make_block
from base.grinder import Grinder
from base.helpers import BOOST, CACHE_FOLDER, Status
from data.serialize import BlockPatternsSerializer


def intervals_points(size_prefix: int, size_postfix: int) -> list[np.float64]:
    points: set[np.float64] = set()
    for adv_cluster in range(1, size_postfix + size_prefix):
        for honest_cluster in range(1, size_prefix + size_postfix):
            if (
                adv_cluster + 1 <= size_postfix
                and adv_cluster + honest_cluster + 1 <= size_prefix + size_postfix
            ):
                point = np.float64(
                    (honest_cluster - BOOST) / (adv_cluster + 2 * honest_cluster)
                )
                points.add(np.round(point, 8))

    points.add(np.float64(0.0))
    points = sorted(list(points))
    return points


def cut(attack_string: str) -> str:
    while attack_string[0] == "h":
        attack_string = attack_string[1:]
    return attack_string


def calc_attack_string_mapping(
    alpha: np.float64, size_prefix: int, size_postfix: int
) -> tuple[dict[str, bool], dict[str, str]]:
    mapping: dict[str, str] = {}
    known: dict[str, bool] = {}

    attack_strings = [
        str(attack_string)
        for attack_string in AttackString.possibilities(
            size_postf_prev=size_postfix, size_pref_next=size_prefix
        )
    ]
    orderer = lambda x: (len(x.split(".")[0]), len(x.split(".")[1]))
    attack_strings = sorted(attack_strings, key=orderer)

    for attack_string in attack_strings:
        before, after = attack_string.split(".")
        if len(before) == 0:
            mapping[attack_string] = "."  # attack_string
            known[attack_string] = True
            continue
        if before[-1] == "h" and after == "":
            mapping[attack_string] = "."
            known[attack_string] = False
            continue
        if before.count("h") == 0:
            mapping[attack_string] = attack_string
            known[attack_string] = True
            continue
        whole = f"{before}{after}"
        adv_cluster = 0
        honest_cluster = 0
        index = 0
        while whole[index] == "a":
            adv_cluster += 1
            index += 1
        while whole[index] == "h":
            honest_cluster += 1
            index += 1
        assert whole[index] == "a"
        next_nofork = f"{before[index:]}.{after}"
        next_fork = f"{before[index+1:]}.{after}"
        if next_fork[0] == "h":  # "AHAHA." => "AHA."
            mapping[attack_string] = mapping[next_nofork]
            known[attack_string] = False
        elif not known[next_fork]:
            mapping[attack_string] = mapping[next_nofork]
            known[attack_string] = False
        elif (honest_cluster - BOOST) / (adv_cluster + 2 * honest_cluster) < alpha:
            mapping[attack_string] = attack_string
            known[attack_string] = True
        else:
            mapping[attack_string] = mapping[next_nofork]
            known[attack_string] = False

    for attack_string in attack_strings:
        before, after = attack_string.split(".")
        if before.count("h") == 0:
            mapping[attack_string] = f"{before}."
        elif len(before) > 0 and before[-1] == "a" and len(after) > 0:
            if after[0] == "h":
                mapping[attack_string] = mapping[attack_string].split(".")[0] + "."
            elif after[0] == "a":
                index = 0
                honest_cluster = 0
                adv_cluster2 = 0
                adv_cluster1 = 0
                while index < len(before) and before[-index - 1] == "a":
                    index += 1
                    adv_cluster2 += 1
                while index < len(before) and before[-index - 1] == "h":
                    index += 1
                    honest_cluster += 1
                while index < len(before) and before[-index - 1] == "a":
                    index += 1
                    adv_cluster1 += 1
                if all(x > 0 for x in [adv_cluster1, honest_cluster, adv_cluster2]):
                    if (adv_cluster2 + honest_cluster - BOOST) / (
                        adv_cluster1 + 2 * (honest_cluster + adv_cluster2)
                    ) >= alpha:
                        mapping[attack_string] = (
                            mapping[attack_string].split(".")[0] + "."
                        )
    assert set(mapping.values()) == set(
        key for key, val in mapping.items() if key == val
    )

    return known, mapping


def calc_prefix_mapping(attack_string_mapping: dict[str, str]) -> dict[str, str]:
    prefixes = set(
        attack_string.split(".")[1] for attack_string in attack_string_mapping
    )
    prefix_to_possib: dict[str, list[str]] = {prefix: [] for prefix in prefixes}

    for attack_string_from, attack_string_to in attack_string_mapping.items():
        prefix_to_possib[attack_string_from.split(".")[1]].append(
            attack_string_to.split(".")[1]
        )

    result: dict[str, str] = {}

    for prefix, possibs in prefix_to_possib.items():
        if all(possibs[0] == possib for possib in possibs):
            result[prefix] = possibs[0]
        else:
            result[prefix] = prefix
    return result


def calc_postfix_mapping(attack_string_mapping: dict[str, str]) -> dict[str, str]:
    postfixes = set(
        attack_string.split(".")[0] for attack_string in attack_string_mapping
    )
    prefixes = set(
        attack_string.split(".")[1] for attack_string in attack_string_mapping
    )
    postfix_to_possib: dict[str, list[str]] = {prefix: [] for prefix in postfixes}

    for attack_string_from, attack_string_to in attack_string_mapping.items():
        postfix_to_possib[attack_string_from.split(".")[0]].append(
            attack_string_to.split(".")[0]
        )

    result: dict[str, str] = {}

    for postfix, possibs in postfix_to_possib.items():
        max_possib = max(possibs, key=len)
        assert postfix.endswith(max_possib)
        for prefix in prefixes:
            attack_string = f"{postfix}.{prefix}"
            max_attack_string = f"{max_possib}.{prefix}"
            assert (
                attack_string_mapping[attack_string]
                == attack_string_mapping[max_attack_string]
            ), f"{attack_string=} {max_attack_string=}\n{attack_string_mapping[attack_string]=} {attack_string_mapping[max_attack_string]=}"
        result[postfix] = max_possib
    return result


def calc_eas_mapping(
    attack_string_mapping: dict[str, str], postfix_mapping: dict[str, str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for attack_string_from, attack_string_to in attack_string_mapping.items():
        for postfix_from, postfix_to in postfix_mapping.items():
            result[f"{attack_string_from}#{postfix_from}"] = (
                f"{attack_string_to}#{postfix_to}"
            )
    return result


class PreEvaluator:
    def __init__(self, alpha: np.float64, size_prefix: int, size_postfix: int):
        self.alpha = alpha
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix

        self.attack_string_to_blocks: dict[str, list[list[BlockData]]] = {}

    def append_selfish_mixing(
        self, blocks: list[list[BlockData]]
    ) -> list[list[BlockData]]:
        missed_possibs = deepcopy(blocks)
        for sequence in missed_possibs:
            slot = min(block.slot for block in sequence) - 1 if sequence else -1
            sequence.insert(
                0,
                make_block(
                    slot=slot,
                    parent_block_slot=None,
                    status=Status.MISSED,
                ),
            )
        for sequence in blocks:
            slot = min(block.slot for block in sequence) - 1 if sequence else -1
            for block in sequence:
                if block.status == Status.PROPOSED:
                    block.parent_block_slot = slot
                    break
            sequence.insert(
                0,
                make_block(
                    slot=slot,
                    parent_block_slot=None,
                    status=Status.PROPOSED,
                ),
            )
        return [*blocks, *missed_possibs]

    def append_n_missed_and_one_proposed_blocks(
        self, blocks: list[list[BlockData]], n: int
    ) -> list[list[BlockData]]:
        assert n >= 0
        result: list[list[BlockData]] = []
        for sequence in blocks:
            for proposed_ind in range(n + 1):
                seq_copy = deepcopy(sequence)
                slot = min(block.slot for block in sequence) - 1
                for i in range(n + 1):
                    new_block = make_block(
                        slot=slot - i,
                        parent_block_slot=None,
                        status=Status.MISSED,
                    )
                    if i == proposed_ind:
                        new_block.status = Status.PROPOSED
                        for block in seq_copy:
                            if block.status == Status.PROPOSED:
                                block.parent_block_slot = slot - i
                                break
                    seq_copy.insert(0, new_block)

                result.append(seq_copy)
        return result

    def append_n_proposed_blocks(
        self, blocks: list[list[BlockData]], n: int
    ) -> list[list[BlockData]]:
        result = deepcopy(blocks)
        for sequence in result:
            for _ in range(n):
                slot = min(block.slot for block in sequence) - 1 if sequence else -1
                new_block = make_block(
                    slot=slot, parent_block_slot=None, status=Status.PROPOSED
                )
                for block in sequence:
                    if block.status == Status.PROPOSED:
                        assert block.parent_block_slot is None, f"{sequence=}"
                        block.parent_block_slot = slot
                        break
                sequence.insert(0, new_block)
        return result

    def append_n_blocks(
        self, blocks: list[list[BlockData]], n: int, status: Status
    ) -> list[list[BlockData]]:
        result = deepcopy(blocks)
        for sequence in result:
            for _ in range(n):
                slot = min(block.slot for block in sequence) - 1 if sequence else -1
                new_block = make_block(slot=slot, parent_block_slot=None, status=status)

                sequence.insert(0, new_block)
        return result

    def eval_attack_string(self, attack_string: str) -> Optional[list[list[BlockData]]]:
        before, after = attack_string.split(".")
        if before.count("h") == 0:
            num_of_adv_slots = before.count("a")
            if num_of_adv_slots == 0:
                if after == "":
                    return [[]]
                n = after.count("h")
                occurance: list[BlockData] = []
                for slot in range(n + 1):
                    occurance.append(
                        make_block(
                            slot=slot,
                            parent_block_slot=slot - 1 if slot > 0 else None,
                            status=Status.PROPOSED,
                        )
                    )
                return [occurance]
            else:
                old_attack_string = f"{before[1:]}.{after}"
                self.attack_string_to_blocks[old_attack_string]
                previous = deepcopy(self.attack_string_to_blocks[old_attack_string])
                return self.append_selfish_mixing(blocks=previous)
        if len(after) == 0 and len(before) > 0 and before[-1] == "h":
            return None
        whole = f"{before}{after}"
        adv_cluster1 = 0
        honest_cluster = 0
        adv_cluster2 = 0
        index = 0
        while whole[index] == "a":
            adv_cluster1 += 1
            index += 1
        while whole[index] == "h":
            honest_cluster += 1
            index += 1
        while index < len(whole) and whole[index] == "a":
            adv_cluster2 += 1
            index += 1
        assert (
            adv_cluster1 > 0 and honest_cluster > 0 and adv_cluster2 > 0
        ), f"{attack_string=}"
        limit = len(before) - (adv_cluster1 + honest_cluster + adv_cluster2)
        _, voting = dynamic_vote(
            alpha=self.alpha,
            adv_cluster1=adv_cluster1,
            honest_cluster=honest_cluster,
            adv_cluster2=adv_cluster2,
            epoch_boundary=limit,
        )
        base_str = f"{before[adv_cluster1+honest_cluster:]}.{after}"
        honest_blocks_before = min(len(before) - adv_cluster1, honest_cluster)
        assert honest_blocks_before > 0

        base_blocks = self.attack_string_to_blocks[base_str]
        # Adding noforking
        result = self.append_n_proposed_blocks(
            blocks=base_blocks,
            n=adv_cluster1 + honest_blocks_before,
        )
        # Adding regret
        regret_base = self.append_n_proposed_blocks(
            blocks=base_blocks,
            n=honest_blocks_before,
        )
        for vote_res in voting:
            regret_cl = self.append_n_blocks(
                blocks=regret_base, n=vote_res.regret_sac, status=Status.MISSED
            )
            diff = adv_cluster1 - vote_res.regret_sac - vote_res.free_slots_before
            assert 0 <= diff <= 1, f"{diff=}"
            if diff == 1:
                regret_cl = self.append_n_proposed_blocks(regret_cl, n=1)
            for _ in range(vote_res.free_slots_before):
                regret_cl = self.append_selfish_mixing(
                    blocks=regret_cl,
                )
            result.extend(regret_cl)
        # Adding forking
        for vote_res in voting:

            for plan in vote_res.plans:

                forking_str = (
                    f"{before[adv_cluster1+honest_cluster+1+plan.addit:]}.{after}"
                )
                if forking_str[0] == "h":
                    continue  # TODO: Consider not skipping this
                assert (
                    plan.addit == 0
                    or len(before) >= adv_cluster1 + honest_cluster + plan.addit
                )
                forking_plan = self.attack_string_to_blocks[forking_str]
                if adv_cluster1 + honest_cluster + plan.addit < len(before):
                    forking_plan = self.append_n_proposed_blocks(
                        blocks=forking_plan,
                        n=1,
                    )
                forking_plan = self.append_n_blocks(
                    blocks=forking_plan,
                    n=plan.addit,
                    status=Status.MISSED,
                )
                forking_plan = self.append_n_blocks(
                    forking_plan, n=honest_blocks_before, status=Status.REORGED
                )
                for sequence in forking_plan:
                    for i in range(honest_cluster):
                        sequence[i].status = Status.REORGED
                        if i == 0:
                            sequence[i].parent_block_slot = (
                                sequence[0].slot
                                - 1
                                - plan.free_slots_after
                                - plan.possibilities_middle
                            )
                        else:
                            sequence[i].parent_block_slot = sequence[i].slot - 1
                for _ in range(plan.free_slots_after):
                    forking_plan = self.append_selfish_mixing(forking_plan)

                forking_plan = self.append_n_missed_and_one_proposed_blocks(
                    blocks=forking_plan, n=plan.possibilities_middle - 1
                )
                diff = (
                    adv_cluster1
                    - plan.free_slots_after
                    - vote_res.free_slots_before
                    - plan.possibilities_middle
                )
                assert 0 <= diff <= 1
                if diff == 1:
                    forking_plan = self.append_n_proposed_blocks(forking_plan, n=1)
                for _ in range(vote_res.free_slots_before):
                    forking_plan = self.append_selfish_mixing(forking_plan)
                result.extend(forking_plan)
        return result

    def eval_all_attack_strings(self) -> dict[str, list[list[BlockData]]]:
        attack_strings = AttackString.possibilities(
            size_pref_next=self.size_prefix, size_postf_prev=self.size_postfix
        )
        ordering = lambda x: (len(x.postfix_prev.postfix), len(x.prefix_next.prefix))
        attack_strings = sorted(attack_strings, key=ordering)
        for attack_string in attack_strings:
            evaled = self.eval_attack_string(attack_string=str(attack_string))
            if evaled is not None:
                self.attack_string_to_blocks[str(attack_string)] = evaled
        return {
            attack_string: self.append_n_proposed_blocks(blocks=blocks, n=1)
            for attack_string, blocks in self.attack_string_to_blocks.items()
        }


class PreEvalGrinder(Grinder[PreEvaledTyp]):
    def __init__(self, out_path: str, size_prefix: int, size_postfix: int):
        super().__init__(
            out_path=out_path,
            default_data={},
            serializer=BlockPatternsSerializer(),
            safety_save=False,
        )
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix

    def _grind(self, **kwargs):
        points = intervals_points(
            size_prefix=self.size_prefix, size_postfix=self.size_postfix
        )
        points.append(points[-1] + 0.1)
        start_and_end = [
            (start, end)
            for start, end in zip(points[:-1], points[1:])
            if float(start) not in self._data
        ]
        for start, end in tqdm(start_and_end, desc="Preeval"):

            alpha = (start + end) / 2
            preevaluator = PreEvaluator(
                alpha=alpha,
                size_prefix=self.size_prefix,
                size_postfix=self.size_postfix,
            )
            mapping = preevaluator.eval_all_attack_strings()
            _, attack_string_mapping = calc_attack_string_mapping(
                alpha=alpha,
                size_prefix=self.size_prefix,
                size_postfix=self.size_postfix,
            )
            newmapping = {
                key: val
                for key, val in mapping.items()
                if key == attack_string_mapping[key]
            }
            self._data_changed = True
            self._data[float(start)] = attack_string_mapping, newmapping


def preeval_grinding(size_prefix: int, size_postfix: int) -> Optional[PreEvaledTyp]:
    grinder = PreEvalGrinder(
        out_path=os.path.join(
            CACHE_FOLDER, "preeval", f"blocks_{size_prefix=}_{size_postfix=}"
        ),
        size_prefix=size_prefix,
        size_postfix=size_postfix,
    )
    return grinder.start_grinding()


class PatternsWrapper:
    def __init__(self, mappings: PreEvaledTyp):
        self.__mappings = mappings
        self.__keys = sorted(list(self.__mappings))

    def __getitem__(
        self, alpha: float
    ) -> tuple[dict[str, str], dict[str, list[list[BlockData]]]]:
        minindex = 0
        maxindex = len(self.__keys) - 1
        while minindex < maxindex:
            med = math.ceil((minindex + maxindex) / 2)
            if self.__keys[med] < alpha:
                minindex = med
            else:
                maxindex = med - 1
        assert minindex == maxindex
        return self.__mappings[self.__keys[minindex]]


if __name__ == "__main__":
    # patterns = preeval_grinding(size_prefix=2, size_postfix=8)
    # patterns_wrapper = PatternsWrapper(mappings=patterns)
    # print(patterns_wrapper[0.271][0]["aaahha."])

    alpha = 0.201
    size_prefix = 2
    size_postfix = 6
    _, attack_string_mapping = calc_attack_string_mapping(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
    )
    eas_mapping = calc_eas_mapping(
        attack_string_mapping=attack_string_mapping,
        postfix_mapping=calc_postfix_mapping(attack_string_mapping),
    )
    print(f"{len(set(eas_mapping.values()))=}")
