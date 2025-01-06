import json
from typing import Any
from base.helpers import SLOTS_PER_EPOCH, Status
from data.file_manager import FileManager
from data.process.make_alternatives import get_active_validators


def collect_anomalies(file_manager: FileManager):
    with open("data/jsons/dune_nft.json", "r") as f:
        data: list[dict[str, Any]] = json.load(f)
    blocks: list[int] = [entry["block_number"] for entry in data]
    beaconchain = file_manager.beaconchain()
    index_to_entity = file_manager.index_to_entity()
    active_validators = get_active_validators(file_manager)
    print("All data read")
    blocks_to_slots = {block.block: slot for slot, block in beaconchain.items()}
    before = 4
    log_lines = []
    for block in blocks:
        slot = blocks_to_slots[block]
        max_slot = (slot // SLOTS_PER_EPOCH - 1) * SLOTS_PER_EPOCH

        flagged = any(
            [
                beaconchain[sl].status != Status.PROPOSED
                for sl in range(max_slot - before, max_slot)
            ]
        )
        if not flagged:
            continue
        indices = active_validators.get_validators(slot // SLOTS_PER_EPOCH)
        entity = index_to_entity[beaconchain[slot].proposer_index]
        stake = sum(index_to_entity[index] == entity for index in indices) / len(
            indices
        )
        log_lines.append(f"Flagged {block=} in epoch={slot // SLOTS_PER_EPOCH}")
        log_lines.append(
            f"Entity that had the block is {entity} with stake {round(100 * stake, 4)}%"
        )
        for sl in range(max_slot - before, max_slot):
            log_lines.append(
                f"slot={sl} status={beaconchain[sl].status.name} entity={index_to_entity[beaconchain[sl].proposer_index]}"
            )
        log_lines.append("\n")
        with open("logs/nft_logs.txt", "w") as f:
            f.write("\n".join(log_lines))


if __name__ == "__main__":
    collect_anomalies(FileManager.file_manager())
