import pytest
from base.beaconchain import BlockData
from base.helpers import MISSING, Status


def test_beaconblock_addition_non_conflicting():
    first = BlockData(
        slot=1,
        proposer_index=MISSING,
        block=0,
        parent_block_slot=None,
        status=Status.MISSED,
        randao_reveal=MISSING,
        randomness=MISSING,
    )
    second = BlockData(
        slot=1,
        proposer_index=3,
        block=0,
        parent_block_slot=MISSING,
        status=MISSING,
        randao_reveal=MISSING,
        randomness="0x12",
    )
    third = BlockData(
        slot=1,
        proposer_index=3,
        block=0,
        parent_block_slot=None,
        status=Status.MISSED,
        randao_reveal=MISSING,
        randomness="0x12",
    )
    assert first + second == third, f"{third=}\n\n{first+second=}\n"
    # Changing only the status to test whether status edge case works correctly
    first.status = Status.MISSED
    second.status = Status.REORGED
    third.status = Status.REORGED
    assert first + second == third, f"{third=}\n\n{first+second=}\n"


def test_beaconblock_addition_conflicting():
    first = BlockData(
        slot=1,
        proposer_index=MISSING,
        block=0,
        parent_block_slot=None,
        status=Status.MISSED,
        randao_reveal=MISSING,
        randomness=MISSING,
    )
    second = BlockData(
        slot=1,
        proposer_index=3,
        block=4,  # conflicting with block=0
        parent_block_slot=MISSING,
        status=MISSING,
        randao_reveal=MISSING,
        randomness="0x12",
    )
    with pytest.raises(AssertionError):
        first + second
    first = BlockData(
        slot=1,
        proposer_index=MISSING,
        block=0,
        parent_block_slot=None,
        status=Status.REORGED,
        randao_reveal=MISSING,
        randomness=MISSING,
    )
    second = BlockData(
        slot=1,
        proposer_index=3,
        block=0,
        parent_block_slot=MISSING,
        status=Status.PROPOSED,
        randao_reveal=MISSING,
        randomness="0x12",
    )
    with pytest.raises(AssertionError):
        first + second


block1 = BlockData(
    slot=0,
    proposer_index=9,
    block=MISSING,
    parent_block_slot=31,
    status=Status.PROPOSED,
    randao_reveal=None,
    randomness="0xabv",
    block_root="0x1234",
    parent_root=MISSING,
)

pattern_block1 = BlockData(
    slot=0,
    proposer_index=MISSING,
    block=MISSING,
    parent_block_slot=-1,
    status=Status.PROPOSED,
    randao_reveal=MISSING,
    randomness=MISSING,
)

pattern_block2 = BlockData(
    slot=0,
    proposer_index=MISSING,
    block=MISSING,
    parent_block_slot=-1,
    status=Status.MISSED,
    randao_reveal=MISSING,
    randomness=MISSING,
)

pattern_block3 = BlockData(
    slot=0,
    proposer_index=MISSING,
    block=MISSING,
    parent_block_slot=None,
    status=Status.PROPOSED,
    randao_reveal=MISSING,
    randomness=MISSING,
)

pattern_block4 = BlockData(
    slot=23,
    proposer_index=MISSING,
    block=MISSING,
    parent_block_slot=None,
    status=Status.PROPOSED,
    randao_reveal=MISSING,
    randomness=MISSING,
)

matching_arguments = [
    (block1, block1),
    (block1, pattern_block1),
    (block1, pattern_block3),
]


@pytest.mark.parametrize("base_block, pattern_block", matching_arguments)
def test_matching_blocks(base_block: BlockData, pattern_block: BlockData):
    assert base_block.is_matching(pattern_block)


non_matching_arguments = [
    (block1, pattern_block2),
    (block1, pattern_block4),
]


@pytest.mark.parametrize("base_block, pattern_block", non_matching_arguments)
def test_non_matching_blocks(base_block: BlockData, pattern_block: BlockData):
    assert not base_block.is_matching(pattern_block)
