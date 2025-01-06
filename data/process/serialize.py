from dataclasses import asdict
import json
from typing import Any
from base.beaconchain import ReorgEntry
from base.serialize import JSON, CustomSerializer
from base.statistics import EpochEntityStat


class EpochEntityStatSerializer(CustomSerializer[dict[int, EpochEntityStat]]):
    def _to_data(self, data: JSON) -> dict[int, EpochEntityStat]:
        return {int(key): EpochEntityStat(**val) for key, val in data.items()}

    def _from_data(self, data: dict[int, EpochEntityStat]) -> JSON:
        return {str(key): asdict(val) for key, val in data.items()}


class ReordEntryEncoder(json.JSONEncoder):
    def default(self, object: Any) -> Any:
        if isinstance(object, ReorgEntry):
            return asdict(object)
        else:
            return super().default(object)


class ReorgEntryDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if all(
            key in dct for key in ["entity", "epoch", "slots", "stake", "block_number"]
        ):
            return ReorgEntry(**dct)
        return dct


class ReorgEntrySerializer(CustomSerializer[dict[int, ReorgEntry]]):
    def __init__(self):
        super().__init__(
            encoder=ReordEntryEncoder,
            decoder=ReorgEntryDecoder,
        )

    def _to_data(self, data: JSON) -> dict[int, ReorgEntry]:
        return {int(key): val for key, val in data.items()}


class IntDictSerializer(CustomSerializer[dict[int, Any]]):
    def _to_data(self, data: JSON) -> dict[int, Any]:
        return {int(key): val for key, val in data.items()}
