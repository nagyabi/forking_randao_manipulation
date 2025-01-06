from copy import deepcopy
from dataclasses import dataclass
import pickle
from typing import Optional

from base.helpers import Comparison
from data.file_manager import FileManager


@dataclass
class CasesEntry:
    outcome: Comparison
    attacker: str
    stake: float
    epoch: int
    attack_string: str
    real_statuses: str
    RO: int
    statuses_to_RO: dict[str, Optional[int]]

    def cut(self, size_postfix: int):
        before, after = self.attack_string.split(".")
        while len(before) > size_postfix:
            before = before[1:]
            self.real_statuses = self.real_statuses[1:]
            self.statuses_to_RO = {
                status[1:]: RO
                for status, RO in self.statuses_to_RO.items()
                if status[0] == "p"
            }
        while len(before) > 0 and before[0] == "h":
            before = before[1:]
            self.real_statuses = self.real_statuses[1:]
            self.statuses_to_RO = {
                status[1:]: RO
                for status, RO in self.statuses_to_RO.items()
                if status[0] == "p"
            }
        self.attack_string = f"{before}.{after}"


@dataclass
class EpochEntityStat:
    """
    In epoch E, we are intrested in the number of proposed blocks by an entity
    and the expected value.
    """

    expected: dict[str, float]
    actual: dict[str, int]


@dataclass
class EpochEntityPossibilityStat:
    num_of_validators: int
    entity_to_number_of_validators: dict[str, int]


def get_index_to_entity(file_manager: FileManager) -> dict[int, str]:
    validators = file_manager.validators()
    deposits = file_manager.deposits()
    address_txn = file_manager.address_txn()
    entity_mapping = file_manager.entities()

    index_to_entity: dict[int, str] = {}
    pub_key_to_txn = {deposit.public_key: deposit.txn_hash for deposit in deposits}
    for index, validator in validators.items():
        txn = pub_key_to_txn.get(validator.public_key[2:])
        if txn is None:
            continue
        address = address_txn.get(txn)
        if address is None:
            continue
        index_to_entity[index] = entity_mapping.get(address, address)

    return index_to_entity


def index_to_entity_job():
    index_to_entity = get_index_to_entity(file_manager=FileManager.file_manager())
    with open("data/processed_data/index_to_entity.pkl", "wb") as f:
        pickle.dump(index_to_entity, f)


def read_delivery_cases(path: str) -> list[CasesEntry]:
    result: list[CasesEntry] = []

    with open(path, "r") as delivery_file:
        lines = delivery_file.readlines()
    lines = [line.strip() for line in lines[1:]]

    for line in lines:
        if line[-1] == ";":
            line = line[:-1]
        spl = line.split(";")
        spl = [s.strip() for s in spl]

        result.append(
            CasesEntry(
                outcome=Comparison[spl[0]],
                attacker=spl[1],
                stake=float(spl[2]),
                epoch=int(spl[3]),
                attack_string=spl[4].lower(),
                real_statuses=spl[5],
                RO=int(spl[6]),
                statuses_to_RO={
                    st: int(RO) if RO != "-" else None
                    for st, RO in zip(spl[7::2], spl[8::2])
                },
            )
        )

    return result


def to_selfish_mixing(data: list[CasesEntry]) -> list[CasesEntry]:
    """
    Filters out every reorg candidate, except for those when the candidate applied 'NOFORK' (most of the times)
    AHA. and ppm will remain, but AHA. and mpm will be filtered out.

    Args:
        data (list[CasesEntry]): data from delivery

    Returns:
        list[CasesEntry]: filtered entries
    """

    result: list[CasesEntry] = []

    for entry in data:
        if entry.attack_string.count("h") == 0:
            result.append(entry)
        else:
            before = entry.attack_string.split(".")[0]
            index_of_last_h = before.rfind("h")
            assert index_of_last_h >= 0, f"{entry=}"
            all_prop = entry.real_statuses[: 1 + index_of_last_h]
            if (
                all_prop.count("p") == 1 + index_of_last_h
                and entry.attack_string[1 + index_of_last_h] == "a"
                and entry.attack_string[: 1 + index_of_last_h].count(".") == 0
            ):
                new_entry = deepcopy(entry)
                new_entry.attack_string = entry.attack_string[1 + index_of_last_h :]
                new_entry.real_statuses = new_entry.real_statuses[1 + index_of_last_h :]
                new_entry.statuses_to_RO = {
                    status[1 + index_of_last_h :]: RO
                    for status, RO in new_entry.statuses_to_RO.items()
                    if status[: 1 + index_of_last_h].count("p") == 1 + index_of_last_h
                }

                result.append(new_entry)
    return result
