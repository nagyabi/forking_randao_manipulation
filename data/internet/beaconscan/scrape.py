import json
import queue
import time
from typing import Optional
import requests

from base.beaconchain import BlockData, Validator
from base.helpers import (
    DEFAULT,
    MISSING,
    RED,
    Status,
    read_headers,
    beachonscan_status_mapping,
)
from data.internet.utils import safe_function

headers = read_headers("data/internet/headers/beaconscan.header")


class TextProcessingHelpers:
    @staticmethod
    def grab_index(text: str) -> int:
        after = text.split("/validator/")[1]
        right = after.split("'")[0]
        return int(right)

    @staticmethod
    def grab_pub_key(text: str) -> str:
        after = text.split("/validator/")[1]
        right = after.split("'")[0]
        assert right[0:2] == "0x" and all(
            c.isdigit() or c in "abcdef" for c in right[2:]
        ), f"Not hexadecimal format: {right=}"
        return right

    @staticmethod
    def grab_balance(text: str) -> int:
        return int(text.split(" ")[0])

    @staticmethod
    def grab_epoch(text: str) -> Optional[int]:
        if text == "--":
            return None
        if "/epoch/" not in text:
            return int(text.split("<")[-2].split(">")[-1])
        after = text.split("/epoch/")[1]
        right = after.split("'")[0]
        return int(right)

    @staticmethod
    def grab_slashed(text: str) -> bool:
        return text == "true"


def validators(page: int, length: int = 100) -> dict[int, Validator]:
    """
    Args:
        page (int): page of the data
        length (int, optional): Length of how many validators will be requested. Defaults to 100 at most 1000 is recommended.

    Returns:
        dict[int, Validator]: Mapping to validators by the validator index to the given validator
    """
    url = f"https://beaconscan.com/datasource?q=validators&type=total&networkId=&sid=614ffd4339340389058b24b75ebbd02f&draw=5&columns%5B0%5D%5Bdata%5D=index&columns%5B0%5D%5Bname%5D=&columns%5B0%5D%5Bsearchable%5D=true&columns%5B0%5D%5Borderable%5D=true&columns%5B0%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B0%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B1%5D%5Bdata%5D=publickey&columns%5B1%5D%5Bname%5D=&columns%5B1%5D%5Bsearchable%5D=true&columns%5B1%5D%5Borderable%5D=false&columns%5B1%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B1%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B2%5D%5Bdata%5D=currentBalance&columns%5B2%5D%5Bname%5D=&columns%5B2%5D%5Bsearchable%5D=true&columns%5B2%5D%5Borderable%5D=true&columns%5B2%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B2%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B3%5D%5Bdata%5D=effectiveBalance&columns%5B3%5D%5Bname%5D=&columns%5B3%5D%5Bsearchable%5D=true&columns%5B3%5D%5Borderable%5D=true&columns%5B3%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B3%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B4%5D%5Bdata%5D=proposed&columns%5B4%5D%5Bname%5D=&columns%5B4%5D%5Bsearchable%5D=true&columns%5B4%5D%5Borderable%5D=true&columns%5B4%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B4%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B5%5D%5Bdata%5D=eligibilityEpoch&columns%5B5%5D%5Bname%5D=&columns%5B5%5D%5Bsearchable%5D=true&columns%5B5%5D%5Borderable%5D=true&columns%5B5%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B5%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B6%5D%5Bdata%5D=activationEpoch&columns%5B6%5D%5Bname%5D=&columns%5B6%5D%5Bsearchable%5D=true&columns%5B6%5D%5Borderable%5D=true&columns%5B6%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B6%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B7%5D%5Bdata%5D=exitEpoch&columns%5B7%5D%5Bname%5D=&columns%5B7%5D%5Bsearchable%5D=true&columns%5B7%5D%5Borderable%5D=true&columns%5B7%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B7%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B8%5D%5Bdata%5D=withEpoch&columns%5B8%5D%5Bname%5D=&columns%5B8%5D%5Bsearchable%5D=true&columns%5B8%5D%5Borderable%5D=true&columns%5B8%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B8%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B9%5D%5Bdata%5D=slashed&columns%5B9%5D%5Bname%5D=&columns%5B9%5D%5Bsearchable%5D=true&columns%5B9%5D%5Borderable%5D=false&columns%5B9%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B9%5D%5Bsearch%5D%5Bregex%5D=false&order%5B0%5D%5Bcolumn%5D=0&order%5B0%5D%5Bdir%5D=asc&start={(page-1)*length}&length={length}&search%5Bvalue%5D=&search%5Bregex%5D=false&_=1716927877840"
    answer = requests.get(url=url, headers=headers)
    text = answer.text
    json_obj = json.loads(text)
    raw_data = json_obj["data"]
    for i, part in enumerate(raw_data):
        assert "index" in part, f"index was not a key of {part=}"
    assert i + 1 == length, f"{page=} {length=} {i=}"
    result: dict[int, Validator] = {}
    for validator_raw_data in raw_data:
        if validator_raw_data["eligibilityEpoch"] == "pending":
            continue
        index = TextProcessingHelpers.grab_index(validator_raw_data["index"])
        if "estimated" in validator_raw_data["activationEpoch"]:
            print(f"{RED}SKIPPING ESTIMATED VALIDATOR:", index, DEFAULT)
            continue
        result[index] = Validator(
            index=index,
            effective_balance=TextProcessingHelpers.grab_balance(
                validator_raw_data["effectiveBalance"]
            ),
            slashed=TextProcessingHelpers.grab_slashed(validator_raw_data["slashed"]),
            public_key=TextProcessingHelpers.grab_pub_key(
                validator_raw_data["publickey"]
            ),
            start_slot=TextProcessingHelpers.grab_epoch(
                validator_raw_data["activationEpoch"]
            ),
            end_slot=TextProcessingHelpers.grab_epoch(validator_raw_data["exitEpoch"]),
        )
    return result


def slot_data(slot: int) -> BlockData:
    """
    Would not recommend using web scraping function in general but there is no API for beaconscan
    """
    url = f"https://beaconscan.com/slot/{slot}"
    answer = requests.get(url=url, headers=headers)
    text = answer.text
    after = text.split('title="Status">')[1]
    after2 = after.split('<div class="col-md-9 js-focus-state ">')[1]
    status = after2.split(">")[1].split("<")[0]
    extracted_data: dict[str] = {}
    data_headers = ["Randao Reveal", "BlockRoot Hash", "ParentRoot Hash"]
    if status != "skipped":
        for header in data_headers:
            after = text.split(f"{header}</h6>")[1]
            after = after.split('<div class="col-md-9 font-size-1">')[1]
            extracted_data[header] = after.split("</div>")[0].strip()
        # Further processing for parent root
        after = text.split("ParentRoot Hash</h6>")[1]
        extracted_data["ParentRoot Hash"] = (
            extracted_data["ParentRoot Hash"]
            .split("/slot?hash=")[1]
            .split(">")[0]
            .strip()
        )
        assert extracted_data["ParentRoot Hash"][-1] in "\"'"
        extracted_data["ParentRoot Hash"] = extracted_data["ParentRoot Hash"][:-1]

    return BlockData(
        slot=slot,
        proposer_index=MISSING,
        block=MISSING,
        parent_block_slot=MISSING,
        status=beachonscan_status_mapping[status],
        randao_reveal=extracted_data.get("Randao Reveal"),
        randomness=MISSING,
        block_root=extracted_data.get("BlockRoot Hash"),
        parent_root=extracted_data.get("ParentRoot Hash"),
    )


def conspiracy(thread_name: str, in_queue: queue.Queue, out_queue: queue.Queue):
    res: dict[int, BlockData] = {}
    try:
        while True:
            slot = in_queue.get()
            if slot == -1:
                break
            print(f"[{thread_name}] getting {slot=}")
            res[slot] = safe_function(slot_data, slot=slot)
            if res[slot].status != Status.MISSED:
                print(f"[{thread_name}] Found not matching data for {slot=}")
            time.sleep(1.0)
    except Exception as e:
        print(f"{RED}[{thread_name}] Exception occured {e}{DEFAULT}")
    out_queue.put(res)


def number_of_validators() -> int:

    url = "https://beaconscan.com/validators"
    answer = requests.get(url=url, headers=headers)
    text = answer.text
    after = text.split("Overview (")[1]
    str_res = after.split(")")[0]
    return int(str_res)
