from dataclasses import asdict
import json
from typing import Any

from base.beaconchain import BlockData, Deposit, PreEvaledTyp, Validator
from base.helpers import MISSING, Status
from base.serialize import JSON, CustomSerializer
from theory.method.quant.base import ForkQuantizedEAS, QuantizedEAS, SMQuantizedEAS
from theory.method.detailed_distribution import Outcome


def deserialize_block(slot_data) -> BlockData:
    return BlockData(
        slot=slot_data["slot"],  # slot is mandatory
        proposer_index=slot_data.get("proposer_index", MISSING),
        block=slot_data.get("block", MISSING),
        parent_block_slot=slot_data.get("parent_block_slot", MISSING),
        status=Status(slot_data["status"]) if "status" in slot_data else MISSING,
        randao_reveal=slot_data.get("randao_reveal", MISSING),
        randomness=slot_data.get("randomness", MISSING),
        block_root=slot_data.get("block_root", MISSING),
        parent_root=slot_data.get("parent_root", MISSING),
    )


def serialize_block(block: BlockData) -> JSON:
    block_dict = asdict(block)
    block_dict["status"] = block.status.value
    block_dict = {
        key: val for key, val in block_dict.items() if block.__dict__[key] != MISSING
    }
    return block_dict


class BeaconBlockEncoder(json.JSONEncoder):
    def default(self, object: Any) -> Any:
        if isinstance(object, BlockData):
            return serialize_block(block=object)
        else:
            return super().default(object)


class BeaconBlockDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "slot" in dct and "status" in dct:
            return deserialize_block(slot_data=dct)
        return dct


class QuantizedEASEndoder(json.JSONEncoder):
    def default(self, object: Any) -> Any:
        if isinstance(object, QuantizedEAS):
            ser = asdict(object)
            del ser["num_of_epoch_strings"]
            del ser["epoch_string_to_index"]
            return ser
        else:
            return super().default(object)


class QuantizedEASDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "id_to_outcome" in dct and "group" in dct:
            return SMQuantizedEAS(
                slot=int(dct["slot"]),
                id_to_outcome={
                    int(key): Outcome.from_dict(val)
                    for key, val in dct["id_to_outcome"].items()
                },
                num_of_epoch_strings=None,
                epoch_string_to_index=None,
                group={int(key): val for key, val in dct["group"].items()},
            )
        elif "id_to_outcome" in dct and "head_group" in dct and "regret_groups" in dct:
            return ForkQuantizedEAS(
                slot=int(dct["slot"]),
                id_to_outcome={
                    int(key): Outcome.from_dict(val)
                    for key, val in dct["id_to_outcome"].items()
                },
                num_of_epoch_strings=None,
                epoch_string_to_index=None,
                head_group={int(key): val for key, val in dct["head_group"].items()},
                regret_groups=[
                    {int(key): val for key, val in group.items()}
                    for group in dct["regret_groups"]
                ],
            )
        return dct


class QuantizedEASSerializer(CustomSerializer[dict[str, QuantizedEAS]]):
    def __init__(self):
        super().__init__(
            encoder=QuantizedEASEndoder,
            decoder=QuantizedEASDecoder,
        )


class BeaconChainSerializer(CustomSerializer[dict[int, BlockData]]):
    def _to_data(self, data: JSON) -> dict[int, BlockData]:
        beaconchain: dict[int, BlockData] = {}
        for str_slot, slot_data in data.items():
            assert slot_data["slot"] == int(str_slot)
            beaconchain[int(str_slot)] = deserialize_block(slot_data=slot_data)
        return beaconchain

    def _from_data(self, data: dict[int, BlockData]) -> JSON:
        return {str(slot): serialize_block(block=block) for slot, block in data.items()}


class BlockPatternSerializer(CustomSerializer[dict[str, list[list[BlockData]]]]):
    def __init__(self):
        super().__init__(
            encoder=BeaconBlockEncoder,
            decoder=BeaconBlockDecoder,
        )


class BlockPatternsSerializer(CustomSerializer[PreEvaledTyp]):
    def __init__(self):
        super().__init__(
            encoder=BeaconBlockEncoder,
            decoder=BeaconBlockDecoder,
        )

    def _to_data(self, data: JSON) -> dict[float, dict[str, list[list[BlockData]]]]:
        return {float(key): val for key, val in data.items()}


class ValidatorSerializer(CustomSerializer[dict[int, Validator]]):
    def _to_data(self, data: JSON) -> dict[int, Validator]:
        return {int(index): Validator(**validator) for index, validator in data.items()}

    def _from_data(self, data: dict[int, Validator]) -> JSON:
        return {str(index): asdict(validator) for index, validator in data.items()}


class DepositSerializer(CustomSerializer[list[Deposit]]):
    def _to_data(self, data: JSON) -> list[Deposit]:
        return [Deposit(**deposit) for deposit in data]

    def _from_data(self, data: list[Deposit]) -> JSON:
        return [asdict(deposit) for deposit in data]
