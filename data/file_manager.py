from dataclasses import dataclass
from typing import Generic, Optional

from base.beaconchain import BlockData, Deposit, Validator
from base.serialize import DataType, IdentitySerializer, PickleSerializer, Serializer
from data.serialize import BeaconChainSerializer, DepositSerializer, ValidatorSerializer


@dataclass
class FileData(Generic[DataType]):
    path: str
    serializer: Serializer[DataType]
    data: Optional[DataType] = None


class FileManager:
    def __init__(
        self,
        beaconchain_path: str = "data/jsons/beaconchain.json",
        validators_path: str = "data/jsons/validators.json",
        deposits_path: str = "data/jsons/deposits.json",
        address_txn_path: str = "data/jsons/address_txn.json",
        entity_mapping_path: str = "data/jsons/entities.json",
        index_to_entity_path: str = "data/processed_data/index_to_entity.pkl",
    ) -> None:
        self._beaconchain: FileData[dict[int, BlockData]] = FileData(
            path=beaconchain_path,
            serializer=BeaconChainSerializer(),
        )
        self._validators: FileData[dict[int, Validator]] = FileData(
            path=validators_path,
            serializer=ValidatorSerializer(),
        )
        self._deposits: FileData[list[Deposit]] = FileData(
            path=deposits_path,
            serializer=DepositSerializer(),
        )
        self._address_txn: FileData[dict[str, str]] = FileData(
            path=address_txn_path,
            serializer=IdentitySerializer(),
        )
        self._entities: FileData[dict[str, str]] = FileData(
            path=entity_mapping_path,
            serializer=IdentitySerializer(),
        )
        self._index_to_entity: FileData[dict[int, str]] = FileData(
            path=index_to_entity_path,
            serializer=PickleSerializer(),
        )

    __file_manager: Optional["FileManager"] = None

    def beaconchain(self) -> dict[int, BlockData]:
        if self._beaconchain.data is None:
            with open(self._beaconchain.path, "r") as f:
                self._beaconchain.data = self._beaconchain.serializer.deserialize(f)
        return self._beaconchain.data

    def validators(self) -> dict[int, Validator]:
        if self._validators.data is None:
            with open(self._validators.path, "r") as f:
                self._validators.data = self._validators.serializer.deserialize(f)
        return self._validators.data

    def deposits(self) -> list[Deposit]:
        if self._deposits.data is None:
            with open(self._deposits.path, "r") as f:
                self._deposits.data = self._deposits.serializer.deserialize(f)
        return self._deposits.data

    def address_txn(self) -> dict[str, str]:
        if self._address_txn.data is None:
            with open(self._address_txn.path, "r") as f:
                self._address_txn.data = self._address_txn.serializer.deserialize(f)
        return self._address_txn.data

    def entities(self) -> dict[str, str]:
        if self._entities.data is None:
            with open(self._entities.path, "r") as f:
                self._entities.data = self._entities.serializer.deserialize(f)
        return self._entities.data

    def index_to_entity(self) -> dict[int, str]:
        if self._index_to_entity.data is None:
            with open(self._index_to_entity.path, "rb") as f:
                self._index_to_entity.data = (
                    self._index_to_entity.serializer.deserialize(f)
                )
        return self._index_to_entity.data

    def file_manager() -> "FileManager":
        if FileManager.__file_manager is None:
            FileManager.__file_manager = FileManager()
        return FileManager.__file_manager
