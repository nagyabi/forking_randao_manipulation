from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
import subprocess
from typing import Any, Optional
import numpy as np


SLOTS_PER_EPOCH = 32
BOOST = 0.4
BIG_ENTITIES = [
    "Lido",
    "Coinbase",
    "Binance",
    "Kraken",
    "Bitcoin Suisse",
    "Upbit",
    "OKX",
    "Celsius",
    "Revolut",
    "CoinSpot",
]
LATEST_DELIVERY = "data/processed_data/2024_10_02_delivery/cases.csv"
CACHE_FOLDER = "cache3"
FIGURES_FOLDER = "Figures"


@dataclass(frozen=True)
class Missing:
    pass


MISSING = Missing()


class Status(Enum):
    PROPOSED = 1
    MISSED = 2
    REORGED = 3

    def __str__(self) -> str:
        return self.name


class Comparison(Enum):
    BEST = 1
    NEUTRAL = 2
    WORSE = 3
    UNKNOWN = 4
    UNKNOWN_WORSE = 5


GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
ORANGE = "\033[38;2;255;165;0m"
DEFAULT = "\033[0m"

beaconcha_status_mapping: dict[str, Status] = {
    "Proposed": Status.PROPOSED,
    "Missed": Status.MISSED,
    "Missed (Orphaned)": Status.REORGED,
}

beachonscan_status_mapping: dict[str, Status] = {
    "proposed": Status.PROPOSED,
    "skipped": Status.MISSED,
    "forked": Status.REORGED,
}

ENDIANNESS = "little"
SIMULATION_SEED = (
    b"ETH_SIM_0"  # should be a const during a simulation, can change between 2
)


def bytes_to_uint64(data: bytes) -> np.uint64:
    """
    Return the integer deserialization of ``data`` interpreted as ``ENDIANNESS``-endian.
    """
    return np.uint64(int.from_bytes(data, ENDIANNESS))


def uint_to_bytes(n: np.uint) -> bytes:
    return n.tobytes(order="C")


def int_to_bytes(n: int) -> bytes:
    return n.to_bytes(length=4, byteorder=ENDIANNESS)


def sha_256(data: bytes) -> bytes:
    sha256_hash = hashlib.sha256(data)
    return sha256_hash.digest()


def read_api_keys(filename: str) -> list[str]:
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []  # No API keys found


def read_headers(filename: str) -> dict[str, str]:
    if not os.path.exists(filename):
        return {}
    with open(filename, "r") as f:
        lines = f.readlines()
    keys = [line[:-2] for line in lines[::2]]
    vals = [line[:-1] if line[-1] == "\n" else line for line in lines[1::2]]
    return {key: val for key, val in zip(keys, vals)}


def git_status() -> tuple[dict[str, Any], Optional[str]]:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        commit_hash = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE
            )
            .stdout.decode("utf-8")
            .strip()
        )
        result = {"commit": commit_hash}
        diff = (
            subprocess.run(["git", "diff"], stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )
        return result, diff
    except subprocess.CalledProcessError:
        return {}, None


def sac(before: str, cfg: int) -> np.ndarray:
    result = 0
    for c, adv_char in zip(bin(cfg)[2:].zfill(len(before))[::-1], before):
        assert c in ["0", "1"]
        assert adv_char in ["a", "h"]
        if c == "0" and adv_char == "a":
            result += 1

    return result
