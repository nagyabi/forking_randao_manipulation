from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from base.helpers import MISSING, SLOTS_PER_EPOCH, Status


Epoch = Optional[int]


@dataclass
class BlockData:
    """
    Block type used for storing actual data.
    """

    slot: int
    proposer_index: int
    block: int
    parent_block_slot: Optional[int]
    # List of child blocks could make it easier to maintain the three, but not every validator has the same view of the three.

    status: Status

    randao_reveal: str
    randomness: (
        str  # xor values of all the previous(current not included) randao_reveals_xored
    )

    block_root: str = MISSING
    parent_root: str = MISSING

    def __add__(self, other: "BlockData") -> "BlockData":
        resdict = {}
        for key, val in self.__dict__.items():
            if val == MISSING:
                resdict[key] = other.__dict__[key]
            else:
                if (
                    key == "status"
                    and self.status != MISSING
                    and other.status != MISSING
                    and (self.status == Status.MISSED or other.status == Status.MISSED)
                ):
                    if self.status == Status.MISSED:
                        resdict[key] = other.status
                    elif other.status == Status.MISSED:
                        resdict[key] = self.status
                else:
                    assert (
                        other.__dict__[key] == MISSING or other.__dict__[key] == val
                    ), f"\n{self=}\n{other=}\n"
                    resdict[key] = val
        return BlockData(**resdict)

    def is_matching(self, pattern: "BlockData") -> bool:
        """_summary_

        Args:
            pattern (BlockData): pattern BlockData to compare if its matching

        Returns:
            bool: whether it's matching pattern
        """
        if self.status != pattern.status:
            return False
        if (self.slot - pattern.slot) % SLOTS_PER_EPOCH != 0:
            return False
        assert pattern.parent_block_slot != MISSING
        if pattern.status == Status.MISSED:
            return True
        if pattern.parent_block_slot is None:
            # Genesis block
            return True
        if self.parent_block_slot is None or self.parent_block_slot == MISSING:
            # If data is not consistent it can skip it, because it's missing
            return False
        return (
            self.parent_block_slot - pattern.parent_block_slot
        ) % SLOTS_PER_EPOCH == 0


@dataclass
class Block:
    parent_block: Optional["Block"]
    block_votes: float
    slot: int
    is_genesis: bool

    # ONLY USED IN HONEST VALIDATOR' get_head_block algorythm
    accumulated: float = None
    head_slot: Optional[int] = None
    children: list["Block"] = None


@dataclass
class Validator:
    index: int
    effective_balance: int
    slashed: bool

    public_key: str

    start_slot: int
    end_slot: Optional[int]


@dataclass(frozen=True)
class Deposit:
    block_number: str
    deposit_amount: int
    public_key: str
    txn_hash: str


@dataclass(frozen=True)
class EpochConfig:
    epoch: int
    last_slots: int
    entities: set[str] = field(default_factory=set)


@dataclass
class EpochAlternatives:
    entity_to_alpha: dict[str, float]
    alternatives: dict[tuple[bool, ...], list[int]]


def make_block(slot: int, parent_block_slot: int, status: Status) -> BlockData:
    return BlockData(
        slot=slot,
        proposer_index=MISSING,
        block=MISSING,
        parent_block_slot=parent_block_slot,
        status=status,
        randao_reveal=MISSING,
        randomness=MISSING,
    )


@dataclass
class ValueDetailed:
    value: np.float64
    scaled: np.float64


@dataclass
class AttackStringEntry:
    RO: ValueDetailed
    adv_slots: ValueDetailed
    forked_honest_blocks: ValueDetailed
    fork_prob: ValueDetailed
    regret_prob: ValueDetailed
    sm_prob: ValueDetailed
    probability: np.float64


PreEvaledTyp = dict[float, tuple[dict[str, str], dict[str, list[list[BlockData]]]]]


@dataclass
class ReorgEntry:
    entity: str
    forked_entitiy: str
    epoch: int
    slots: int
    stake: float
    block_number: int
