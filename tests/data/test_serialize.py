import os
import tempfile
import pytest

from base.beaconchain import BlockData, Deposit, ReorgEntry, Validator
from base.helpers import MISSING, Status
from base.serialize import Serializer
from data.process.serialize import ReorgEntrySerializer
from data.serialize import (
    BeaconChainSerializer,
    BlockPatternSerializer,
    DepositSerializer,
    ValidatorSerializer,
)

beaconchain1: dict[int, BlockData] = {
    1: BlockData(
        slot=1,
        proposer_index=4,
        block=MISSING,
        parent_block_slot=0,
        status=Status.PROPOSED,
        randao_reveal="0x12",
        randomness=MISSING,
        block_root="0x482942",
        parent_root=None,
    ),
    2: BlockData(
        slot=2,
        proposer_index=MISSING,
        block=3,
        parent_block_slot=MISSING,
        status=Status.MISSED,
        randao_reveal=None,
        randomness="0xabc",
        block_root=MISSING,
        parent_root="0x334",
    ),
}

beaconchain2: dict[int, BlockData] = {
    10: BlockData(
        slot=10,
        proposer_index=43,
        block=-1,
        parent_block_slot=MISSING,
        status=Status.PROPOSED,
        randao_reveal=MISSING,
        randomness=MISSING,
        parent_root="0x52",
        block_root="xyz",
    ),
}

validators1: dict[int, Validator] = {
    4: Validator(
        index=4,
        effective_balance=42556,
        slashed=False,
        public_key="0x465267",
        start_slot=552,
        end_slot=798,
    ),
    5: Validator(
        index=5,
        effective_balance=4412556,
        slashed=True,
        public_key="0x46a62",
        start_slot=1552,
        end_slot=None,
    ),
}

validators2: dict[int, Validator] = {
    0: Validator(
        index=0,
        effective_balance=88,
        slashed=True,
        public_key="0xabcde",
        start_slot=5,
        end_slot=89,
    )
}

deposits: list[Deposit] = [
    Deposit(
        block_number=31,
        deposit_amount=None,
        public_key="0x75ae",
        txn_hash="7d2",
    ),
    Deposit(
        block_number=89,
        deposit_amount=47892,
        public_key="0xabb",
        txn_hash="014",
    ),
]

block_pattern1: dict[str, list[list[BlockData]]] = {
    "a123": [
        [
            BlockData(
                slot=33,
                proposer_index=22,
                block=MISSING,
                parent_block_slot=-3,
                status=Status.MISSED,
                randao_reveal=None,
                randomness=MISSING,
                block_root="ag",
                parent_root="0x",
            ),
            BlockData(
                slot=-4,
                proposer_index=MISSING,
                block=5,
                parent_block_slot=-3,
                status=Status.PROPOSED,
                randao_reveal="aaa",
                randomness="0xy",
                block_root="",
                parent_root=MISSING,
            ),
        ]
    ],
    "a1": [[]],
    "xyz": [
        [
            BlockData(
                slot=4455,
                proposer_index=1,
                block=10,
                parent_block_slot=3,
                status=Status.REORGED,
                randao_reveal="ae",
                randomness=None,
                block_root=MISSING,
                parent_root="empty",
            )
        ]
    ],
}

reorg_entries = {
    1236: ReorgEntry(
        entity="Lido",
        forked_entitiy="0x1232",
        epoch="1450",
        slots=422,
        stake=0.98,
        block_number=738914,
    ),
    14: ReorgEntry(
        entity="Kraken",
        forked_entitiy="Lido",
        epoch="898",
        slots=-12,
        stake=0.23,
        block_number=-12,
    ),
    22: ReorgEntry(
        entity="Lido",
        forked_entitiy="Coinbase",
        epoch="1450310",
        slots=33,
        stake=0.15,
        block_number=47821,
    ),
}

arguments = [
    (BeaconChainSerializer(), beaconchain1),
    (BeaconChainSerializer(), beaconchain2),
    (ValidatorSerializer(), validators1),
    (ValidatorSerializer(), validators2),
    (DepositSerializer(), deposits),
    (BlockPatternSerializer(), block_pattern1),
    (ReorgEntrySerializer(), reorg_entries),
]


@pytest.mark.parametrize("serializer, data", arguments)
def test_beaconchain_serialization(serializer: Serializer, data):
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "data.json")
        with open(filename, "w") as file:
            serializer.serialize(data=data, file=file)
        with open(filename, "r") as file:
            deserialized = serializer.deserialize(file)
        assert deserialized == data, f"{data=}\n{deserialized}"
