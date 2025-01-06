import json
import queue
import time
from typing import Any, Optional
import requests
from base.beaconchain import Deposit
from base.helpers import read_api_keys
from data.internet.utils import APITester, safe_function


apikeys = read_api_keys("data/internet/apikeys/etherscan.json")


def process_data(data: str) -> dict[str, Any]:
    deposit_bytes = bytes.fromhex(data[706:722])
    deposit = int.from_bytes(deposit_bytes, byteorder="little")
    signature = data[386:482]
    return {"deposit": deposit, "public_key": signature}


def get_fromaddress_from_txn(txn: str, api_key: str) -> str:
    """
    Returns address from the txn hash
    """
    url = f"https://api.etherscan.io/api?module=proxy&action=eth_getTransactionByHash&txhash={txn}&apikey={api_key}"
    answer = requests.get(url=url)
    text = answer.text
    data = json.loads(text)
    data = data["result"]
    assert data["hash"] == txn, f"Error for {txn=}"
    return data["from"]


def get_deposits(
    block: int, page: int, api_key: str, max_block: int = 19506335
) -> Optional[list[Deposit]]:
    url = f"https://api.etherscan.io/api?module=logs&action=getLogs&address=0x00000000219ab540356cBB839Cbe05303d7705Fa&fromBlock={block}&toBlock={max_block}&page={page}&topic0=0x649bbc62d0e31342afea4e5cd82d4049e7e1ee912fc0889aa790803be39038c5&offset=1000&apikey={api_key}"
    answer = requests.get(url=url)
    response = json.loads(answer.text)

    if response["message"] != "OK":
        return None
    data = response["result"]

    res: list[Deposit] = []
    for block in data:
        processed_block = process_data(data=block["data"])
        res.append(
            Deposit(
                block_number=block["blockNumber"],
                deposit_amount=processed_block["deposit"],
                public_key=processed_block["public_key"],
                txn_hash=block["transactionHash"],
            )
        )
    return res


def addresses_from_txns(
    txns: list[str], api_key: str, thread_num: int, out_queue: queue.Queue
):
    res = {}
    try:
        for txn in txns:
            print(f"[{thread_num}] Grinding {txn=}..")
            newdata = safe_function(get_fromaddress_from_txn, txn=txn, api_key=api_key)
            res[txn] = newdata

            time.sleep(0.2)
    except Exception:
        pass
    out_queue.put(res)


class EtherscanAPITester(APITester):
    def test_api_keys(self, api_keys: list[str]) -> dict[str, bool]:
        given_input = (
            "0x0b53e18a1460fc029a4114a3ef56a633abcae57b8240213f18f04cb4dd59073f"
        )
        expected_output = "0xed15c84f822ecad56dbf6bc922c9c5fed2e2e8c4"
        result: dict[str, bool] = {}
        for api_key in api_keys:
            try:
                output = get_fromaddress_from_txn(txn=given_input, api_key=api_key)
                result[api_key] = expected_output == output
            except Exception:
                result[api_key] = False
            time.sleep(0.2)
        return result


def MEV_from_block(block: int, api_key: str) -> Optional[int]:
    tag = hex(block)
    url_number = f"https://api.etherscan.io/api?module=proxy&action=eth_getBlockTransactionCountByNumber&tag={tag}&apikey={api_key}"
    answer_number = requests.get(url=url_number)
    data_number = json.loads(answer_number.text)

    tx_num = int(data_number["result"], 16) - 1
    if tx_num < 0:
        return None
    index_hex = hex(tx_num)
    url_transaction = f"https://api.etherscan.io/api?module=proxy&action=eth_getTransactionByBlockNumberAndIndex&tag={tag}&index={index_hex}&apikey={api_key}"
    # time.sleep(0.1)
    answer_txn = requests.get(url=url_transaction)
    data_txn = json.loads(answer_txn.text)
    value = int(data_txn["result"]["value"], 16)
    return value


def MEVs_from_blocks(
    blocks: list[int], api_key: str, thread_num: int, out_queue: queue.Queue
) -> None:
    result: dict[int, Optional[int]] = {}

    exc_counter = 0
    for i, block in enumerate(blocks):
        try:
            value = safe_function(MEV_from_block, block=block, api_key=api_key)
            result[block] = value
            print(
                f"[{thread_num}] Getting {block=} - {round(100 * i / len(blocks), 3)} %"
            )
        except Exception:
            exc_counter += 1
            print(f"[{thread_num}] Exception occured for {block=} {exc_counter} times")

    out_queue.put(result)


if __name__ == "__main__":
    value = MEV_from_block(17552147, apikeys[0])
    print(f"{value=}")
