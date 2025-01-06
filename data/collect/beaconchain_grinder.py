import os
import queue
import threading
import time
from typing import Optional
from base.beaconchain import BlockData
from base.grinder import Grinder
from base.helpers import DEFAULT, MISSING, RED, Status
from base.shuffle import SLOTS_PER_EPOCH
from data.serialize import BeaconChainSerializer
from data.internet.beaconcha.api import BeaconChaAPIConnection
from data.internet.connection import BeaconChainConnection
from data.internet.beaconscan.scrape import conspiracy


class BeaconChainGrinder(Grinder[dict[int, BlockData]]):
    def __init__(self, connection: BeaconChainConnection, out_path: str):
        self.connection = connection
        super().__init__(
            out_path=out_path,
            default_data={},
            serializer=BeaconChainSerializer(),
            safety_save=True,
            stack=3,
        )
        self.root_to_slot: dict[str, int]

    def _grind(self, **kwargs):
        self.root_to_slot = {
            block.block_root: slot
            for slot, block in self._data.items()
            if block.block_root != MISSING and block.block_root is not None
        }

        epochs: list[int]

        if "epochs" in kwargs:
            epochs = kwargs["epochs"]
        else:
            max_epoch = kwargs["max_epoch"]  # One of these params are mandatory
            min_epoch = kwargs.get("min_epoch")
            min_epoch = (
                min_epoch if min_epoch is not None else 171104
            )  # default is 2023 jan
            data_slots = self._data.keys()
            data_epochs = [slot // SLOTS_PER_EPOCH for slot in data_slots]
            min_epochs = max([min_epoch, *data_epochs]) + 1
            epochs = list(range(min_epochs, max_epoch + 1))
        print(f"There are {len(epochs)} to gather")
        guarantee_for_no_new_data = "epochs" not in kwargs

        self.conspiracy_in_queue = queue.Queue()
        self.conspiracy_out_queue = queue.Queue()
        self.threads = [
            threading.Thread(
                target=conspiracy,
                args=[
                    "beaconscan",
                    self.conspiracy_in_queue,
                    self.conspiracy_out_queue,
                ],
            ),
        ]
        for thread in self.threads:
            thread.start()

        self._data_changed = len(epochs) > 0
        for epoch in epochs:
            print(f"[main] {epoch=}")
            new_epoch_data = self.connection.epoch_data(epoch=epoch)
            for slot, block in new_epoch_data.items():
                if block.block_root is not None and block.block_root != MISSING:
                    self.root_to_slot[block.block_root] = slot

            for slot, block in new_epoch_data.items():
                if (
                    block.parent_root is not None
                    and block.parent_root != MISSING
                    and block.parent_root in self.root_to_slot
                ):
                    block.parent_block_slot = int(self.root_to_slot[block.parent_root])

            for slot, block in new_epoch_data.items():
                if not guarantee_for_no_new_data and slot in self._data:
                    self._data[slot] += block
                else:
                    self._data[slot] = block
                if block.status == Status.MISSED and block.randao_reveal == MISSING:
                    self.conspiracy_in_queue.put(block.slot)

            time.sleep(self.connection.get_waiting_time())

    def _before_end(self):
        print("Before updating")
        self.conspiracy_in_queue.put(-1)
        for thread in self.threads:
            thread.join()

        if not self.conspiracy_out_queue.empty():
            conspiracy: dict[int, BlockData] = self.conspiracy_out_queue.get()
            for slot, block in conspiracy.items():
                if block.block_root is not None and block.block_root != MISSING:
                    self.root_to_slot[block.block_root] = slot
            for slot, block in conspiracy.items():
                if (
                    block.parent_root is not None
                    and block.parent_root != MISSING
                    and block.parent_root in self.root_to_slot
                ):
                    block.parent_block_slot = int(self.root_to_slot[block.parent_root])

            for slot, block in conspiracy.items():
                if block.status == Status.PROPOSED:
                    changed = True
                    block.status = Status.REORGED
                    if (
                        (slot - 1) in self._data
                        and (slot + 1) in self._data
                        and self._data[slot - 1].status == Status.PROPOSED
                        and self._data[slot + 1].status == Status.PROPOSED
                    ):
                        block1, block2 = (
                            self._data[slot - 1].block,
                            self._data[slot + 1].block,
                        )
                        if isinstance(block1, int) and isinstance(block2, int):
                            if block1 + 1 != block2:
                                block.status = Status.PROPOSED
                                changed = False

                    os.makedirs("logs", exist_ok=True)
                    with open("logs/conflicts_between_explorers.txt", "a") as f:
                        f.write(f"{slot=}\n")
                        f.write(f"{changed=}\n")
                        f.write(f"[beaconcha.in]: {self._data[slot]}\n")
                        f.write(f"[beaconscan.com]: {block}\n\n")

                self._data[slot] = self._data[slot] + block

        else:
            print(f"{RED}conspiracy was empty{DEFAULT}")


def grind_beaconchain(max_epoch: int, min_epoch: Optional[int] = None) -> None:
    print(f"{min_epoch=}")
    connections: list[BeaconChainConnection] = [BeaconChaAPIConnection()]
    available_connections = [
        connection for connection in connections if connection.is_available()
    ]
    assert (
        len(available_connections) > 0
    ), "No available connections to beaconcha.in. Did you forgot to get API keys?"
    connection = available_connections[0]
    grinder = BeaconChainGrinder(
        connection=connection, out_path="data/jsons/beaconchain.json"
    )
    grinder.start_grinding(max_epoch=max_epoch, min_epoch=min_epoch)
