import queue
import threading
import time
from typing import Optional

from tqdm import tqdm
from base.beaconchain import Deposit, Validator
from base.grinder import Grinder
from base.helpers import read_api_keys
from base.serialize import IdentitySerializer
from data.serialize import DepositSerializer, ValidatorSerializer
from data.internet.beaconscan.scrape import number_of_validators, validators
from data.internet.etherscan.api import addresses_from_txns, get_deposits
from data.internet.utils import safe_function


class ValidatorGrinder(Grinder[dict[int, Validator]]):
    """
    This class uses beaconscan.com's validator dataset. This is technically webscraping, but it uses an API like
    call that returns almost clear data, which the website uses to construct the html, we skip that part so its more
    energy efficient.

    We need to regenerate the data every time as we do not monitor the withdrawals/slashings, this rule generally
    applies only to the active validators but we will regerate the whole dataset as alomst every chunk of the
    incoming data consists
    """

    def __init__(self, out_path: str, regenerate: bool, length: int):

        super().__init__(
            out_path=out_path,
            default_data={},
            serializer=ValidatorSerializer(),
            regenerate=regenerate,
            safety_save=True,
            stack=3,
        )
        self.length = length
        self.start_page = (
            1 if len(self._data) == 0 else (max(self._data) + 1) // length + 1
        )

    def _grind(self, **kwargs):

        max_validator: int
        if "max_validator" in kwargs:
            max_validator = kwargs["max_validator"]
        else:
            max_validator = number_of_validators()
            time.sleep(1.0)
        page_amout = max_validator // self.length

        for page in tqdm(
            range(self.start_page, page_amout + 1),
            desc="validators",
            initial=self.start_page,
            total=page_amout,
        ):
            self._data_changed = True
            new_validator_data = safe_function(
                validators, page=page, length=self.length
            )
            self._data.update(new_validator_data)
            time.sleep(1.0)


class DepositGrinder(Grinder[list[Deposit]]):
    def __init__(self, out_path: str, top_block: int):
        super().__init__(
            out_path=out_path,
            default_data=[],
            serializer=DepositSerializer(),
            regenerate=False,
            safety_save=True,
            stack=3,
        )
        self.block = self.maxblock(self._data)
        self.page = 1
        self.top_block = top_block
        self.apikeys = read_api_keys("data/internet/apikeys/etherscan.json")
        self.period = 0

    def _grind(self, **kwargs):
        while self.block <= self.top_block:
            """
            Grinding till we reach top_block
            """
            assert (
                len(self.apikeys) > 0
            ), "No API keys found for etherscan.io.\nRun --config-api-keys etherscan --action help\nThen -config-api-keys etherscan --action append --key-values <APIKey1>"
            print(f"block={self.block} page={self.page}")
            new_data = safe_function(
                function=get_deposits,
                block=self.block,
                page=self.page,
                api_key=self.apikeys[self.period],
                max_block=self.top_block,
            )
            if new_data is None:
                break
            self._data_changed = True
            self._data.extend(new_data)

            self.page += 1
            if self.page > 10:
                self.page = 1
                self.block = max(self.block, self.maxblock(new_data))
            self.period += 1
            self.period %= len(self.apikeys)

    def maxblock(self, data: list[Deposit]) -> int:
        return max([0, *[int(block.block_number[2:], 16) for block in data]])


class AddressGrinder(Grinder[dict[str, str]]):
    def __init__(self, out_path: str):
        super().__init__(
            out_path=out_path,
            default_data={},
            serializer=IdentitySerializer(),
            regenerate=False,
            safety_save=True,
            stack=3,
        )
        self.apikeys = read_api_keys("data/internet/apikeys/etherscan.json")
        assert (
            len(self.apikeys) > 0
        ), "No API keys found for etherscan.io.\nRun --config-api-keys etherscan --action help\nThen -config-api-keys etherscan --action append --key-values <APIKey1>"

    def _grind(self, **kwargs):
        txns = kwargs["txns"]
        txns = list(set([txn for txn in txns if txn not in self._data]))
        print(f"There are {len(txns)} to get the address for")
        self._data_changed = len(txns) > 0
        n = len(self.apikeys)
        # Start clustering the txns among len(self.apikeys) threads to different lists
        thread_txn_length = (len(txns) + n - 1) // n
        thread_txns = [
            txns[(i * thread_txn_length) : ((i + 1) * thread_txn_length)]
            for i in range(n)
        ]
        assert n * thread_txn_length >= len(txns)

        out_queue = queue.Queue()

        threads = [
            threading.Thread(
                target=addresses_from_txns, args=[data, api_key, i + 1, out_queue]
            )
            for i, (data, api_key) in enumerate(zip(thread_txns, self.apikeys))
        ]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        while not out_queue.empty():
            produced_data = out_queue.get()
            self._data.update(produced_data)


def validator_grinding(regenerate: bool, max_validator: Optional[int] = None):
    val_grinder = ValidatorGrinder(
        out_path="data/jsons/validators.json", regenerate=regenerate, length=100
    )
    if max_validator is not None:
        val_grinder.start_grinding(max_validator=max_validator)
    else:
        val_grinder.start_grinding()


def deposit_grinding(top_block: int):
    deposit_grinder = DepositGrinder(
        out_path="data/jsons/deposits.json", top_block=top_block
    )
    deposit_grinder.start_grinding()


def address_grinding():
    with open("data/jsons/deposits.json", "r") as f:
        deposits = DepositSerializer().deserialize(f)
    txns = [deposit.txn_hash for deposit in deposits]
    address_grinder = AddressGrinder(out_path="data/jsons/address_txn.json")
    address_grinder.start_grinding(txns=txns)
