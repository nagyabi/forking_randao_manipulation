from dataclasses import MISSING
from base.shuffle import mix_randao
from data.serialize import BeaconChainSerializer


def test_mix_randao():
    serializer = BeaconChainSerializer()
    with open("tests/data/small_chain", "r") as f:
        small_chain = serializer.deserialize(f)
    for slot, block in small_chain.items():
        if all(
            [
                block.randomness is not None,
                block.randomness != MISSING,
                block.parent_block_slot != MISSING,
            ]
        ):
            if block.parent_block_slot not in small_chain:
                continue
            previous_randomness = bytes.fromhex(
                small_chain[block.parent_block_slot].randomness[2:]
            )
            previous_randao_reveal = small_chain[block.parent_block_slot].randao_reveal[
                2:
            ]
            expected_randomness = bytes.fromhex(block.randomness[2:])
            randomness = mix_randao(
                randomness=previous_randomness, randao_reveal=previous_randao_reveal
            )
            assert expected_randomness == randomness
