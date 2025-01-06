import numpy as np
import pytest

from theory.method.preeval import PreEvaluator
from base.beaconchain import BlockData
from base.helpers import Status

arguments_cons = [
    (np.float64(0.121), 1, 3),
    (np.float64(0.161), 2, 7),
    (np.float64(0.201), 3, 5),
    (np.float64(0.221), 2, 6),
    (np.float64(0.321), 2, 8),
    (np.float64(0.111), 3, 6),
]


def sequence_to_int(seq: list[BlockData]) -> int:
    result = 0
    for block in seq:
        if block.slot >= 0:
            break
        result += 1 if block.status == Status.PROPOSED else 0
        result *= 2
    return result


@pytest.mark.parametrize("alpha, size_prefix, size_postfix", arguments_cons)
def test_preeval_consistency(alpha: np.float64, size_prefix: int, size_postfix: int):
    preevaluator = PreEvaluator(
        alpha=alpha, size_prefix=size_prefix, size_postfix=size_postfix
    )
    mapping = preevaluator.eval_all_attack_strings()
    for attack_string, blocks in mapping.items():
        id_to_seq: dict[int, list[BlockData]] = {}
        for sequence in blocks:
            id = sequence_to_int(sequence)
            assert id not in id_to_seq
            id_to_seq[id] = sequence
            assert len(sequence) == len(attack_string), f"{attack_string=}\n{sequence=}"
            before, _ = attack_string.split(".")
            assert sequence[0].slot == -len(before) - 1
            for prev, block in zip(sequence[:-1], sequence[1:]):
                assert prev.slot + 1 == block.slot
            latest_proposed = -len(before) - 1
            slot_to_block = {block.slot: block for block in sequence}
            for block in sequence[1:]:
                if block.status == Status.PROPOSED:
                    assert (
                        slot_to_block[block.parent_block_slot].status == Status.PROPOSED
                    ), f"{attack_string=}\n{sequence=}"
                    assert block.parent_block_slot == latest_proposed
                    latest_proposed = block.slot
                elif block.status == Status.REORGED:
                    assert slot_to_block[block.parent_block_slot].status in [
                        Status.PROPOSED,
                        Status.REORGED,
                    ]
                elif block.status == Status.MISSED:
                    assert block.parent_block_slot is None
                else:
                    raise ValueError(f"Unknown status: {block.status}")
