from dataclasses import dataclass
from base.helpers import SLOTS_PER_EPOCH
from data.file_manager import FileManager


@dataclass
class EpochAttackStringAndCandidate:
    epoch: int
    attack_string: str
    candidate: str


class DataToAttackString:
    def __init__(
        self, file_manager: FileManager, size_prefix: int, size_postfix: int
    ) -> None:
        self.beaconchain = file_manager.beaconchain()
        self.index_to_entity = file_manager.index_to_entity()
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix

    def to_attack_string_epochs(
        self, epoch: int
    ) -> list[EpochAttackStringAndCandidate]:
        next_epoch_slot = (epoch + 1) * SLOTS_PER_EPOCH
        slots = list(
            range(
                next_epoch_slot - self.size_postfix, next_epoch_slot + self.size_prefix
            )
        )
        if not all(slot in self.beaconchain for slot in slots):
            return []
        entities = [
            self.index_to_entity[self.beaconchain[slot].proposer_index]
            for slot in slots
        ]
        result: list[EpochAttackStringAndCandidate] = []
        for config in range(
            2 ** (self.size_postfix - 1), 2 ** (self.size_prefix + self.size_postfix)
        ):
            candidate: str
            conf_copy = config
            for entity in entities:
                if conf_copy % 2 == 1:
                    candidate = entity
                    break
                conf_copy //= 2
            assert candidate is not None
            matching = True
            attack_string = ""
            after = False
            need_writing = True
            for i, entity in enumerate(entities):
                if config % 2 == 1:
                    if need_writing:
                        attack_string += "a"
                    if entity != candidate:
                        matching = False
                        break
                    if after:
                        need_writing = False
                else:
                    if need_writing:
                        attack_string += "h"
                    if candidate == entity:
                        matching = False
                        break

                if i + 1 == self.size_postfix:
                    attack_string += "."
                    after = True

                config //= 2
            assert candidate is not None

            if matching:
                while attack_string[0] == "h":
                    attack_string = attack_string[1:]
                while attack_string[-1] == "h":
                    attack_string = attack_string[:-1]

                if attack_string[0] != ".":
                    result.append(
                        EpochAttackStringAndCandidate(
                            epoch=epoch,
                            attack_string=attack_string,
                            candidate=candidate,
                        )
                    )
        return result

    def query(self) -> dict[int, list[EpochAttackStringAndCandidate]]:
        min_epoch = min(self.beaconchain) // SLOTS_PER_EPOCH
        max_epoch = max(self.beaconchain) // SLOTS_PER_EPOCH
        result: dict[int, list[EpochAttackStringAndCandidate]] = {}
        for epoch in range(min_epoch, max_epoch + 1):
            part = self.to_attack_string_epochs(epoch=epoch)
            if len(part) > 0:
                result[epoch] = part
        return result
