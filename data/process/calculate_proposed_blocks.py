from copy import deepcopy
import math
import os
import pickle
import tempfile
from base.beaconchain import Validator
from base.grinder import Grinder
from base.helpers import SLOTS_PER_EPOCH, Status
from base.serialize import PickleSerializer
from base.statistics import EpochEntityPossibilityStat, EpochEntityStat
from data.file_manager import FileManager
from data.process.make_alternatives import BIG_EPOCH, MAX_EFFECTIVE_BALANCE


class ActiveEntityValidators:
    """
    Class responsible for computing how much validators does the entity has in a given epoch
    WARNING: Works correctly only when queried with monoton growing series of epochs
    """

    def __init__(
        self,
        validators: dict[int, Validator],
        index_to_entity: dict[int, str],
        entities: list[str],
    ):
        validators_and_entity = [
            (validator, index_to_entity[index])
            for index, validator in validators.items()
        ]
        sorted_by_start = sorted(validators_and_entity, key=lambda x: x[0].start_slot)
        sorted_by_end = sorted(
            validators_and_entity,
            key=lambda x: (x[0].end_slot if x[0].end_slot is not None else BIG_EPOCH),
        )

        self.start_entities = [
            validator_entity[1] for validator_entity in sorted_by_start
        ]
        self.start_slots = [
            validator_entity[0].start_slot for validator_entity in sorted_by_start
        ]

        self.end_entities = [validator_entity[1] for validator_entity in sorted_by_end]
        self.end_slots = [
            (
                validator_entity[0].end_slot
                if validator_entity[0].end_slot is not None
                else BIG_EPOCH
            )
            for validator_entity in sorted_by_end
        ]

        self.data: dict[str, int] = {entity: 0 for entity in entities}
        self.num_of_active_validators: int = 0

        self.start_index = 0
        self.end_index = 0

        # self.index_to_entity = index_to_entity

    def active_entity_validators(self, epoch: int) -> EpochEntityPossibilityStat:
        while (
            self.start_index < len(self.start_slots)
            and self.start_slots[self.start_index] <= epoch
        ):
            self.num_of_active_validators += 1
            if self.start_entities[self.start_index] in self.data:
                self.data[self.start_entities[self.start_index]] += 1
            self.start_index += 1
        while (
            self.end_index < len(self.end_slots)
            and self.end_slots[self.end_index] <= epoch
        ):
            self.num_of_active_validators -= 1
            if self.end_entities[self.end_index] in self.data:
                self.data[self.end_entities[self.end_index]] -= 1
            self.end_index += 1

        return EpochEntityPossibilityStat(
            num_of_validators=self.num_of_active_validators,
            entity_to_number_of_validators=deepcopy(self.data),
        )


class ProposedBlockStatisticsGrinder(Grinder[dict[int, EpochEntityStat]]):
    def __init__(
        self,
        out_path: str,
        file_manager: FileManager,
        index_to_entity: dict[int, str],
        entities: list[str],
    ):
        super().__init__(
            out_path,
            default_data={},
            serializer=PickleSerializer(),
            regenerate=False,
            safety_save=True,
            stack=2,
        )
        self.beaconchain = file_manager.beaconchain()
        self.index_to_entity = index_to_entity
        self.index_to_validator: dict[int, Validator] = {}

        deposits = file_manager.deposits()
        unique_deposits = set(deposits)

        pub_key_to_deposit_amout: dict[str, int] = {}

        for deposit in unique_deposits:
            if deposit.public_key not in pub_key_to_deposit_amout:
                pub_key_to_deposit_amout[deposit.public_key] = 0
            pub_key_to_deposit_amout[deposit.public_key] += deposit.deposit_amount
            if pub_key_to_deposit_amout[deposit.public_key] > MAX_EFFECTIVE_BALANCE:
                pub_key_to_deposit_amout[deposit.public_key] = MAX_EFFECTIVE_BALANCE

        pub_key_to_index: dict[str, int] = {
            validator.public_key[2:]: index
            for index, validator in file_manager.validators().items()
        }

        for pub_key, deposit_amount in pub_key_to_deposit_amout.items():
            if pub_key in pub_key_to_index:
                index = pub_key_to_index[pub_key]
                original_validator = file_manager.validators()[index]
                original_validator.effective_balance = deposit_amount
                if original_validator.effective_balance == MAX_EFFECTIVE_BALANCE:
                    self.index_to_validator[index] = original_validator

        self.active_validators = ActiveEntityValidators(
            validators=file_manager.validators(),
            index_to_entity=self.index_to_entity,
            entities=entities,
        )
        self.entities = entities

    def _grind(self, **kwargs):
        epochs: list[int] = kwargs["epochs"]
        epochs = sorted(list(set(epochs)))

        epochs = [epoch for epoch in epochs if epoch not in self._data]
        self._data_changed = len(epochs) > 0

        for i, epoch in enumerate(epochs):
            print(f"{i} / {len(epochs)}")

            possible_validators = self.active_validators.active_entity_validators(
                epoch=epoch
            )
            actual_validators = [
                self.beaconchain[slot].proposer_index
                for slot in range(
                    epoch * SLOTS_PER_EPOCH, (epoch + 1) * SLOTS_PER_EPOCH
                )
                if self.beaconchain[slot].status == Status.PROPOSED
            ]

            actual: dict[str, int] = {entity: 0 for entity in self.entities}
            for proposer_index in actual_validators:
                entity = self.index_to_entity[proposer_index]
                if entity in self.entities:
                    actual[entity] += 1

            new_data = EpochEntityStat(
                expected={
                    entity: validator_count / possible_validators.num_of_validators
                    for entity, validator_count in possible_validators.entity_to_number_of_validators.items()
                },
                actual=actual,
            )
            self._data[epoch] = new_data


def test():
    index_to_entity = FileManager.file_manager().index_to_entity()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "temp.json")
        grinder = ProposedBlockStatisticsGrinder(
            out_path=temp_file,
            file_manager=FileManager.file_manager(),
            index_to_entity=index_to_entity,
            entities=["Lido", "Kraken"],
        )
        epochs_and_expected: list[tuple[int, int]] = [
            (163411, 479490),
            (193411, 563359),
            (213411, 653635),
            (243411, 884514),
            (253411, 902882),
            (263411, 943575),
            (283411, 1014558),
        ]
        epochs_and_expected = sorted(epochs_and_expected, key=lambda x: x[0])

        for epoch, expected in epochs_and_expected:
            print(f"{epoch=}")
            print(f"{expected=}")

            res = grinder.active_validators.active_entity_validators(epoch=epoch)
            print(f"code {res=}")


def proposer_blocks_job(min_stake: float = 0.003):
    index_to_entities = FileManager.file_manager().index_to_entity()
    beaconchain = FileManager.file_manager().beaconchain()
    stat: dict[str, int] = {}
    min_epoch = min(beaconchain) // SLOTS_PER_EPOCH
    max_epoch = math.ceil((max(beaconchain) + 1) / SLOTS_PER_EPOCH)

    all_slots = 0
    for block in beaconchain.values():
        all_slots += 1
        entity = index_to_entities[block.proposer_index]
        if entity not in stat:
            stat[entity] = 0
        stat[entity] += 1

    entities = [
        entity for entity, amount in stat.items() if amount / all_slots >= min_stake
    ]

    grinder = ProposedBlockStatisticsGrinder(
        out_path="data/processed_data/proposed_blocks_stats.pkl",
        file_manager=FileManager.file_manager(),
        index_to_entity=FileManager.file_manager().index_to_entity(),
        entities=entities,
    )
    grinder.start_grinding(epochs=list(range(min_epoch, max_epoch)))


def watch():
    with open("data/processed_data/proposed_blocks_stats.pkl", "rb") as f:
        data: dict[int, EpochEntityStat] = pickle.load(f)
    for epoch in range(283417, 283467):
        print(f"{epoch=}")
        print("expected:", data[epoch].expected["Kraken"])
        print("actual:", data[epoch].actual["Kraken"])
        print()


if __name__ == "__main__":
    watch()
