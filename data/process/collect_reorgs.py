from tqdm import tqdm
from base.beaconchain import ReorgEntry
from base.grinder import Grinder
from base.helpers import SLOTS_PER_EPOCH, Status
from data.file_manager import FileManager
from data.process.make_alternatives import get_active_validators
from data.process.serialize import ReorgEntrySerializer


class ReorgGrinder(Grinder[dict[int, ReorgEntry]]):
    def __init__(self, out_path: str, file_manager: FileManager):
        super().__init__(
            out_path,
            default_data={},
            serializer=ReorgEntrySerializer(),
            regenerate=False,
            safety_save=False,
        )

        self.beaconchain = file_manager.beaconchain()
        self.index_to_entity = file_manager.index_to_entity()
        self.active_validators = get_active_validators(file_manager)

    def _grind(self, **kwargs):
        min_epoch = min(self.beaconchain) // SLOTS_PER_EPOCH
        max_epoch = max(self.beaconchain) // SLOTS_PER_EPOCH - 2
        from_epoch = 1 + max([min_epoch - 1, *list(self._data)])
        print(f"{from_epoch=}")
        print(f"{min_epoch=}")
        for epoch in tqdm(range(from_epoch, max_epoch)):
            slot = (epoch + 1) * SLOTS_PER_EPOCH
            candidates = [slot, slot - 1]
            for candidate_slot in candidates:
                if (
                    self.beaconchain[candidate_slot - 1].status == Status.REORGED
                    and self.beaconchain[candidate_slot].status == Status.PROPOSED
                ):
                    entity = self.index_to_entity[
                        self.beaconchain[candidate_slot].proposer_index
                    ]
                    forked_entity = self.index_to_entity[
                        self.beaconchain[candidate_slot - 1].proposer_index
                    ]
                    if entity != forked_entity:
                        proposer_indices = self.active_validators.get_validators(
                            epoch=epoch + 2
                        )
                        stake = len(
                            [
                                index
                                for index in proposer_indices
                                if entity == self.index_to_entity[index]
                            ]
                        ) / len(proposer_indices)
                        adv_slots = sum(
                            entity
                            == self.index_to_entity[self.beaconchain[sl].proposer_index]
                            for sl in range(
                                (epoch + 2) * SLOTS_PER_EPOCH,
                                (epoch + 3) * SLOTS_PER_EPOCH,
                            )
                        )
                        self._data_changed = True
                        self._data[epoch] = ReorgEntry(
                            entity=entity,
                            forked_entitiy=forked_entity,
                            epoch=epoch,
                            slots=adv_slots,
                            stake=stake,
                            block_number=self.beaconchain[candidate_slot].block,
                        )


if __name__ == "__main__":
    grinder = ReorgGrinder(
        out_path="data/processed_data/reorgs.json",
        file_manager=FileManager.file_manager(),
    )
    grinder.start_grinding()
