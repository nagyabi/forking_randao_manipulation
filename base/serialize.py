from abc import ABC, abstractmethod
import json
import pickle
from typing import Any, Generic, TypeVar

JSON = dict[str, Any] | list

DataType = TypeVar("DataType")


class Serializer(ABC, Generic[DataType]):
    @abstractmethod
    def serialize(self, data: DataType, file):
        pass

    @abstractmethod
    def deserialize(self, file) -> DataType:
        pass

    def is_binary(self) -> bool:
        return False


class CustomSerializer(Serializer[DataType]):
    def __init__(self, encoder=None, decoder=None):
        self.encoder = encoder
        self.decoder = decoder

    def serialize(self, data: DataType, file):
        json_data = self._from_data(data=data)
        json.dump(json_data, file, cls=self.encoder)

    def deserialize(self, file) -> DataType:
        return self._to_data(data=json.load(file, cls=self.decoder))

    def _from_data(self, data: DataType) -> JSON:
        return data

    def _to_data(self, data: JSON) -> DataType:
        return data


class IdentitySerializer(CustomSerializer[JSON]):
    def _from_data(self, data: JSON) -> JSON:
        return data

    def _to_data(self, data: JSON) -> JSON:
        return data


class PickleSerializer(Serializer[DataType]):
    """
    Only for custom use-cases: dictionary like object but
    we can't dump it into JSON, because it has tuple keys.
    """

    def serialize(self, data: DataType, file):
        pickle.dump(data, file)

    def deserialize(self, file) -> DataType:
        return pickle.load(file)

    def is_binary(self) -> bool:
        return True
