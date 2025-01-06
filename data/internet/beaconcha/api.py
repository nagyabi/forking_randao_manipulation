import json
import time
from typing import Optional
import requests
from base.beaconchain import BlockData
from base.helpers import MISSING, Status, read_api_keys
from data.internet.connection import BeaconChainConnection
from data.internet.utils import APITester


class BeaconChaAPIConnection(BeaconChainConnection):
    def __init__(self, api_keys: Optional[list[str]] = None):
        if api_keys is not None:
            self.apikeys = api_keys
        else:
            self.apikeys = read_api_keys("data/internet/apikeys/beaconcha.json")

    def is_available(self) -> bool:
        return len(self.apikeys) > 0

    def slot_data(self, slot: int) -> BlockData:
        """
        Uses API of beaconcha.in, generally recommended. For Missed slots you won't get any additional data,
        including proposer index which makes sense even for Missed slots.
        """
        assert (
            self.is_available()
        ), f"Expected to have 1 api key, instead got {len(self.apikeys)}"
        api_key = self.apikeys[0]
        url = f"https://beaconcha.in/api/v1/slot/{slot}?apikey={api_key}"
        answer = requests.get(url=url)
        text = answer.text
        data = json.loads(text)
        if (
            data["status"] == "ERROR: could not retrieve db results"
        ):  # Here we are dealing with a Missed Slot
            return BlockData(
                slot=slot,
                proposer_index=MISSING,
                block=MISSING,
                parent_block_slot=MISSING,
                status=Status.MISSED,
                randao_reveal=MISSING,
                randomness=MISSING,
            )
        assert data["status"] == "OK", f"For {slot=} we got status: " + data["status"]
        data = data["data"]
        assert slot == data["slot"]
        status = Status(int(data["status"]))
        return BlockData(
            slot=slot,
            proposer_index=data["proposer"],
            block=data["exec_block_number"],
            parent_block_slot=MISSING,
            status=status,
            randao_reveal=data["randaoreveal"],
            randomness=(
                data["exec_random"][2:] if status == Status.PROPOSED else MISSING
            ),
            block_root=data["blockroot"],
            parent_root=data["parentroot"],
        )

    def epoch_data(self, epoch: int) -> dict[int, BlockData]:
        assert (
            self.is_available()
        ), f"Expected to have 1 api key, instead got {len(self.apikeys)}"
        api_key = self.apikeys[0]
        url = f"https://beaconcha.in/api/v1/epoch/{epoch}/slots?apikey={api_key}"
        answer = requests.get(url=url)
        text = answer.text

        data = json.loads(text)
        assert data["status"] == "OK", (
            f"For {epoch=}, API called with status=" + data["status"]
        )

        data = data["data"]

        result: dict[int, BlockData] = {}

        def to_missing(data, status):
            return data if status != Status.MISSED else MISSING

        for json_block in data:
            slot = json_block["slot"]
            status = Status(int(json_block["status"]))
            randomness = (
                json_block["exec_random"] if status == Status.PROPOSED else None
            )
            randomness = randomness[2:] if randomness is not None else None
            result[slot] = BlockData(
                slot=slot,
                proposer_index=json_block["proposer"],
                block=to_missing(json_block["exec_block_number"], status=status),
                parent_block_slot=MISSING,
                status=status,
                randao_reveal=to_missing(json_block["randaoreveal"], status=status),
                randomness=randomness,
                block_root=to_missing(json_block["blockroot"], status=status),
                parent_root=to_missing(json_block["parentroot"], status=status),
            )

            if result[slot].status == Status.MISSED:
                result[slot].randao_reveal = (
                    MISSING  # This might be a reorged or even a proposed block, this way it won't cause problems merging the new block data
                )

        return result

    def latest_epoch(self) -> int:
        apikey = self.apikeys[0]
        url = f"https://beaconcha.in/api/v1/epoch/latest?apikey={apikey}"
        answer = requests.get(url=url)
        text = answer.text
        data = json.loads(text)

        assert data["status"] == "OK", f"Status is not OK: " + data["status"]
        return int(data["data"]["epoch"])

    def get_waiting_time(self) -> float:
        return 0.2


class BeaconChaAPITester(APITester):
    def test_api_keys(self, api_keys: list[str]) -> dict[str, bool]:
        result: dict[str, bool] = {}

        epoch = 263411
        slot = 8429183
        partial_expected_output = BlockData(
            slot=8429183,
            block=19231189,
            proposer_index=996682,
            parent_block_slot=MISSING,
            status=Status.PROPOSED,
            randao_reveal="0xa71b6b9e366fa1eec17ff2a4a3fd8192b62f02c5a4dcc3a34f9d9d18d957ac1acf548ea7f6d5a1fb6af9029c799792e617aecf6dac5869afb6b5c4a9f404d4a41e711898c75c7d94fdaefda50968489cc07ed888f6a4b1425f4535ad467ad6a0",
            randomness="203f80841748755372b45898fc165345fe86526f75e9c982250cc73dceeaf22d",
            block_root="0x65459c6edc4d58eddbb23ba1e2bf7d004b9c52c8d40f90531f360346568e8779",
            parent_root="0x3f26b6a1b3df0ddc96cbd46214016daa01a676c1d08f23a1914025983e8a2afb",
        )

        for api_key in api_keys:
            connection = BeaconChaAPIConnection(api_keys=[api_key])
            try:
                output = connection.epoch_data(epoch=epoch)
                result[api_key] = output[slot] == partial_expected_output
            except Exception:
                result[api_key] = False
            time.sleep(0.5)
        return result
