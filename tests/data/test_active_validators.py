from typing import Optional
import pytest
from base.beaconchain import Validator
from base.statistics import EpochEntityPossibilityStat
from data.process.calculate_proposed_blocks import ActiveEntityValidators
from data.process.make_alternatives import ActiveValidators


def dummy_validator(index: int, start_slot: int, end_slot: Optional[int]) -> Validator:
    return Validator(
        index=index,
        effective_balance=None,
        slashed=None,
        public_key=None,
        start_slot=start_slot,
        end_slot=end_slot,
    )


validators = {
    0: dummy_validator(index=0, start_slot=0, end_slot=None),
    1: dummy_validator(index=1, start_slot=0, end_slot=6),
    2: dummy_validator(index=2, start_slot=2, end_slot=65),
    3: dummy_validator(index=3, start_slot=30, end_slot=35),
    40: dummy_validator(index=40, start_slot=31, end_slot=None),
    41: dummy_validator(index=41, start_slot=60, end_slot=80),
    42: dummy_validator(index=42, start_slot=70, end_slot=None),
}


arguments_av = [
    (
        validators,
        [-1, 0, 6],
        [
            [],  # -1
            [0, 1],  # 0
            [0, 2],  # 6
        ],
    ),
    (
        validators,
        [2, 5, 6, 22, 30, 35, 69, 71, 888],
        [
            [0, 1, 2],  # 2
            [0, 1, 2],  # 5
            [0, 2],  # 6
            [0, 2],  # 22
            [0, 2, 3],  # 30
            [0, 2, 40],  # 35
            [0, 40, 41],  # 69
            [0, 40, 41, 42],  # 71
            [0, 40, 42],  # 888
        ],
    ),
]


@pytest.mark.parametrize("validators, epochs, expected_validators", arguments_av)
def test_active_validators(
    validators: dict[int, Validator],
    epochs: list[int],
    expected_validators: list[list[int]],
):
    active_validators = ActiveValidators(validators=validators)

    assert sorted(epochs) == epochs, f"Test invalid, epochs should be monoton growing"
    for epoch, expected in zip(epochs, expected_validators):
        result = active_validators.get_validators(epoch=epoch)
        assert result == expected


index_to_entity = {
    0: "Lido",
    1: "Kraken",
    2: "Coinbase",
    3: "Lido",
    40: "0xabc",
    41: "Kraken",
    42: "Lido",
}

arguments_aev = [
    (
        validators,
        index_to_entity,
        ["Lido"],
        [-1, 0, 32],
        [
            EpochEntityPossibilityStat(  # -1
                num_of_validators=0,
                entity_to_number_of_validators={"Lido": 0},
            ),
            EpochEntityPossibilityStat(  # 0
                num_of_validators=2,
                entity_to_number_of_validators={"Lido": 1},
            ),
            EpochEntityPossibilityStat(  # 32
                num_of_validators=4,
                entity_to_number_of_validators={"Lido": 2},
            ),
        ],
    ),
    (
        validators,
        index_to_entity,
        ["Lido", "Kraken", "Coinbase", "0xabc"],
        [2, 5, 6, 22, 30, 35, 69, 71, 888],
        [
            EpochEntityPossibilityStat(  # 2
                num_of_validators=3,
                entity_to_number_of_validators={
                    "Lido": 1,
                    "Kraken": 1,
                    "Coinbase": 1,
                    "0xabc": 0,
                },
            ),
            EpochEntityPossibilityStat(  # 5
                num_of_validators=3,
                entity_to_number_of_validators={
                    "Lido": 1,
                    "Kraken": 1,
                    "Coinbase": 1,
                    "0xabc": 0,
                },
            ),
            EpochEntityPossibilityStat(  # 6
                num_of_validators=2,
                entity_to_number_of_validators={
                    "Lido": 1,
                    "Kraken": 0,
                    "Coinbase": 1,
                    "0xabc": 0,
                },
            ),
            EpochEntityPossibilityStat(  # 22
                num_of_validators=2,
                entity_to_number_of_validators={
                    "Lido": 1,
                    "Kraken": 0,
                    "Coinbase": 1,
                    "0xabc": 0,
                },
            ),
            EpochEntityPossibilityStat(  # 30
                num_of_validators=3,
                entity_to_number_of_validators={
                    "Lido": 2,
                    "Kraken": 0,
                    "Coinbase": 1,
                    "0xabc": 0,
                },
            ),
            EpochEntityPossibilityStat(  # 35
                num_of_validators=3,
                entity_to_number_of_validators={
                    "Lido": 1,
                    "Kraken": 0,
                    "Coinbase": 1,
                    "0xabc": 1,
                },
            ),
            EpochEntityPossibilityStat(  # 69
                num_of_validators=3,
                entity_to_number_of_validators={
                    "Lido": 1,
                    "Kraken": 1,
                    "Coinbase": 0,
                    "0xabc": 1,
                },
            ),
            EpochEntityPossibilityStat(  # 71
                num_of_validators=4,
                entity_to_number_of_validators={
                    "Lido": 2,
                    "Kraken": 1,
                    "Coinbase": 0,
                    "0xabc": 1,
                },
            ),
            EpochEntityPossibilityStat(  # 88
                num_of_validators=3,
                entity_to_number_of_validators={
                    "Lido": 2,
                    "Kraken": 0,
                    "Coinbase": 0,
                    "0xabc": 1,
                },
            ),
        ],
    ),
]


@pytest.mark.parametrize(
    "validators, index_to_entity, entities, epochs, expected_results", arguments_aev
)
def test_active_entity_validators(
    validators: dict[int, Validator],
    index_to_entity: dict[int, str],
    entities: list[str],
    epochs: list[int],
    expected_results: list[EpochEntityPossibilityStat],
):
    active_entity_validators = ActiveEntityValidators(
        validators=validators,
        index_to_entity=index_to_entity,
        entities=entities,
    )

    for epoch, expected in zip(epochs, expected_results):
        actual = active_entity_validators.active_entity_validators(epoch=epoch)
        assert actual == expected
