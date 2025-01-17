from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from base.helpers import BOOST
from theory.method.detailed_distribution import DetailedSlot, DetailedStatus


class BeaconChainMove(Enum):
    PROPOSE = "PROPOSE"
    VOTE = "VOTE"


@dataclass
class BeaconChainAction(ABC):
    slot: int  # Slot where action is executed at


@dataclass
class Vote(BeaconChainAction):
    target_slot: int  # Slot to vote for


@dataclass
class Propose(BeaconChainAction):
    block_slot: int  # Slot where block is located
    parent_block_slot: int  # Slot of head block


class AttackEngine:
    """
    Responsible for computing the corresponding basic actions to an attack
    """

    def __init__(self, head_slot: int, alpha: float):
        self.alpha = alpha
        self.private_head_slot = self.head_slot = head_slot
        self.adv_votes = self.honest_votes = 0.0
        self.withheld_blocks: list[int] = []
        self.silent_fork: bool = (
            True  # True if no forking is happening, otherwise denotes whether the honest validators vote for the
        )

    def propose(self, slot: int) -> list[BeaconChainAction]:
        """
        Proposing a block on self.private_head
        """

        actions: list[BeaconChainAction] = []
        if self.head_slot != self.private_head_slot:  # Finishing forking
            assert (
                self.adv_votes + BOOST > self.honest_votes
            ), f"Not sufficient votes to reorg {self.adv_votes=} {self.honest_votes=}"
            self.adv_votes = self.honest_votes = 0
            slots_with = [*self.withheld_blocks, slot]

            for last_slot, block_slot in zip(slots_with[:-1], slots_with[1:]):
                actions.append(
                    Propose(
                        slot=slot, block_slot=block_slot, parent_block_slot=last_slot
                    )
                )
            self.withheld_blocks = []
        else:
            actions.append(
                Propose(slot=slot, block_slot=slot, parent_block_slot=self.head_slot)
            )

        self.head_slot = self.private_head_slot = slot
        self.silent_fork = True
        actions.append(Vote(slot=slot, target_slot=slot))
        return actions

    def honest_propose(self, slot: int) -> list[BeaconChainAction]:
        if self.head_slot != self.private_head_slot:  # Forking
            self.silent_fork = False
            self.adv_votes += self.alpha
            self.honest_votes += 1 - self.alpha
        else:
            self.private_head_slot = slot
        self.head_slot = slot
        return [Vote(slot=slot, target_slot=self.private_head_slot)]

    def miss(self, slot: int) -> list[BeaconChainAction]:
        if self.head_slot != self.private_head_slot:
            self.adv_votes += self.alpha
            if not self.silent_fork:
                self.honest_votes += 1 - self.alpha
        return [Vote(slot=slot, target_slot=self.private_head_slot)]

    def private_build(self, slot: int) -> list[BeaconChainAction]:
        assert self.silent_fork, f"{slot=}"
        if len(self.withheld_blocks) == 0:  # just starting the attack
            assert self.head_slot == self.private_head_slot
            self.withheld_blocks = [self.head_slot]
        self.withheld_blocks.append(slot)
        self.private_head_slot = slot
        self.adv_votes += self.alpha
        return [Vote(slot=slot, target_slot=slot)]

    def regret(self) -> None:
        assert self.head_slot != self.private_head_slot
        assert not self.silent_fork
        self.adv_votes = self.honest_votes = 0
        self.private_head_slot = self.head_slot
        self.silent_fork = True
        self.withheld_blocks = []

    def feed(self, statuses: Iterable[DetailedSlot]) -> list[BeaconChainAction]:
        collected: list[BeaconChainAction] = []
        for status in statuses:
            match status.status:
                case DetailedStatus.PRIVATE:
                    collected.extend(self.private_build(status.slot))
                case DetailedStatus.MISSED:
                    collected.extend(self.miss(status.slot))
                case DetailedStatus.PROPOSED:
                    collected.extend(self.propose(status.slot))
                case DetailedStatus.REORGED:
                    collected.extend(self.honest_propose(status.slot))
                case DetailedStatus.HONESTPROPOSE:
                    collected.extend(self.honest_propose(status.slot))
        return collected
