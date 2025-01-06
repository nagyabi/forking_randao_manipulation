import queue
import threading
from typing import Optional
from base.grinder import Grinder
from base.helpers import read_api_keys
from base.serialize import PickleSerializer
from data.internet.etherscan.api import MEVs_from_blocks
from data.process.serialize import ReorgEntrySerializer


class MEVGrinder(Grinder[dict[int, Optional[int]]]):
    def __init__(self, out_path: str):
        super().__init__(
            out_path=out_path,
            default_data={},
            serializer=PickleSerializer(),
            regenerate=False,
            safety_save=True,
            stack=1,
        )
        self.apikeys = read_api_keys("data/internet/apikeys/etherscan.json")

    def _grind(self, **kwargs):
        assert (
            len(self.apikeys) > 0
        ), "No API keys found for etherscan.io.\nRun --config-api-keys etherscan --action help\nThen -config-api-keys etherscan --action append --key-values <APIKey1>"
        blocks = kwargs["blocks"]
        blocks = [block for block in blocks if block not in self._data]

        n = len(self.apikeys)
        thread_txn_length = (len(blocks) + n - 1) // n
        thread_txns = [
            blocks[(i * thread_txn_length) : ((i + 1) * thread_txn_length)]
            for i in range(n)
        ]
        assert n * thread_txn_length >= len(blocks)

        out_queue = queue.Queue()
        threads = [
            threading.Thread(
                target=MEVs_from_blocks, args=[data, api_key, i + 1, out_queue]
            )
            for i, (data, api_key) in enumerate(zip(thread_txns, self.apikeys))
        ]
        for thread in threads:
            thread.start()

        self._data_changed = len(blocks) > 0

        for thread in threads:
            thread.join()

        while not out_queue.empty():
            produced_data = out_queue.get()
            self._data.update(produced_data)


def MEV_grinding():
    serializer = ReorgEntrySerializer()
    with open("data/processed_data/reorgs.json", "r") as f:
        reorg_data = serializer.deserialize(f)
    blocks = [entry.block_number for entry in reorg_data.values()]
    grinder = MEVGrinder(out_path="data/jsons/mev.pkl")
    grinder.start_grinding(blocks=blocks)


if __name__ == "__main__":
    MEV_grinding()
