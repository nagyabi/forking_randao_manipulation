import binascii
from typing import Optional

from tqdm import tqdm
from theory.method.preeval import PatternsWrapper, preeval_grinding
from base.beaconchain import EpochAlternatives, Validator
from base.grinder import Grinder
from base.helpers import BLUE, DEFAULT, MISSING, Status
from base.serialize import PickleSerializer
from base.shuffle import SLOTS_PER_EPOCH, generate_epoch, mix_randao
from data.file_manager import FileManager
from data.process.data_to_string import (
    DataToAttackString,
    EpochAttackStringAndCandidate,
)

MAX_EFFECTIVE_BALANCE = SLOTS_PER_EPOCH * 1_000_000_000
BIG_EPOCH = 999999


class ActiveValidators:
    def __init__(self, validators: dict[int, Validator]):
        validators_by_start = sorted(validators.values(), key=lambda x: x.start_slot)
        validators_by_end = sorted(
            validators.values(),
            key=lambda validator: (
                validator.end_slot if validator.end_slot is not None else BIG_EPOCH
            ),
        )
        self.start_indices = [validator.index for validator in validators_by_start]
        self.end_indices = [validator.index for validator in validators_by_end]

        self.start_slots = [validator.start_slot for validator in validators_by_start]
        self.end_slots = [
            validator.end_slot if validator.end_slot is not None else BIG_EPOCH
            for validator in validators_by_end
        ]

        self.start_index = 0
        self.end_index = 0

    def get_validators(self, epoch: int) -> list[int]:
        while (
            self.start_index < len(self.start_slots)
            and self.start_slots[self.start_index] <= epoch
        ):
            self.start_index += 1
        while (
            self.end_index < len(self.end_slots)
            and self.end_slots[self.end_index] <= epoch
        ):
            self.end_index += 1

        set_diff = set(self.start_indices[: self.start_index]) - set(
            self.end_indices[: self.end_index]
        )
        return sorted(list(set_diff))


def get_active_validators(file_manager: FileManager) -> ActiveValidators:
    unique_deposits = set(file_manager.deposits())

    pub_key_to_deposit_amout: dict[str, int] = {}

    for deposit in unique_deposits:
        if deposit.public_key not in pub_key_to_deposit_amout:
            pub_key_to_deposit_amout[deposit.public_key] = 0
        pub_key_to_deposit_amout[deposit.public_key] += deposit.deposit_amount
        if pub_key_to_deposit_amout[deposit.public_key] > MAX_EFFECTIVE_BALANCE:
            pub_key_to_deposit_amout[deposit.public_key] = MAX_EFFECTIVE_BALANCE

    validators = file_manager.validators()

    pub_key_to_index: dict[str, int] = {
        validator.public_key[2:]: index for index, validator in validators.items()
    }

    index_to_validator: dict[int, Validator] = {}

    for pub_key, deposit_amount in pub_key_to_deposit_amout.items():
        if pub_key in pub_key_to_index:
            index = pub_key_to_index[pub_key]
            original_validator = validators[index]
            original_validator.effective_balance = deposit_amount
            if original_validator.effective_balance == MAX_EFFECTIVE_BALANCE:
                index_to_validator[index] = original_validator

    return ActiveValidators(validators=index_to_validator)


class AlternativesProducer(Grinder[dict[int, EpochAlternatives]]):
    def __init__(
        self,
        out_path: str,
        file_manager: FileManager,
        patterns_wrapper: PatternsWrapper,
    ) -> None:
        super().__init__(
            out_path=out_path,
            default_data={},
            serializer=PickleSerializer(),
            regenerate=False,
            safety_save=True,
            stack=2,
        )
        self.index_to_entity = file_manager.index_to_entity()
        self.patterns_wrapper = patterns_wrapper
        # Merging validators and deposits together
        self.active_validators = get_active_validators(file_manager)

        # Reading beaconchain

        self.beaconchain = file_manager.beaconchain()

    def get_validators_fast(self, epoch: int) -> list[int]:
        return self.active_validators.get_validators(epoch=epoch)

    def get_randomness(self, epoch: int) -> Optional[bytes]:
        slot = (epoch - 1) * SLOTS_PER_EPOCH
        while (
            slot in self.beaconchain
            and self.beaconchain[slot].status != Status.PROPOSED
        ):
            slot += 1
        if slot not in self.beaconchain:
            return None
        randomness = self.beaconchain[slot].randomness
        if randomness is None or randomness == MISSING:
            return None
        randomness = randomness[2:] if randomness[:2] == "0x" else randomness
        return bytes.fromhex(randomness)

    def compute_proposer_indices(
        self, epoch: int, indices: list[int], additional_randao_reveals: list[str]
    ) -> tuple[list[int], str]:

        randomness = self.get_randomness(epoch=epoch)
        # print(f"{additional_randao_reveals=}")
        for additional_randao in additional_randao_reveals:
            randomness = mix_randao(
                randomness=randomness, randao_reveal=additional_randao
            )
        epochs = generate_epoch(
            mix=randomness,
            indices=indices,
            epoch=epoch,
            from_slot=epoch * SLOTS_PER_EPOCH,
        )
        randomness_string = binascii.hexlify(randomness).decode()
        return epochs, randomness_string

    def actual_proposer_indices(self, epoch: int) -> list[int]:
        from_slot = SLOTS_PER_EPOCH * epoch
        return [
            self.beaconchain[slot].proposer_index
            for slot in range(from_slot, from_slot + SLOTS_PER_EPOCH)
        ]

    def _grind(self, **kwargs):
        self._data_changed = True  # TODO

        attack_string_query: dict[int, list[EpochAttackStringAndCandidate]] = kwargs[
            "attack_string_query"
        ]

        epochs = sorted(list(attack_string_query))
        for epoch in tqdm(epochs, desc="Alternatives"):
            if self.get_randomness(epoch=epoch + 2) is None:
                print(f"{BLUE}MISSING RANDOMNESS FOR EPOCH={epoch}{DEFAULT}")
                continue

            attack_string_possibs = attack_string_query[epoch]

            entry = (
                self._data[epoch]
                if epoch in self._data
                else EpochAlternatives(entity_to_alpha={}, alternatives={})
            )
            proposer_indices: Optional[list[int]] = None
            entity_to_attack_string: dict[str, str] = {}
            for possib in attack_string_possibs:
                if possib.candidate not in entry.entity_to_alpha:
                    if proposer_indices is None:
                        proposer_indices = self.get_validators_fast(epoch=epoch + 2)
                    entry.entity_to_alpha[possib.candidate] = len(
                        [
                            index
                            for index in proposer_indices
                            if possib.candidate == self.index_to_entity[index]
                        ]
                    ) / len(proposer_indices)
                alpha = entry.entity_to_alpha[possib.candidate]
                attack_string_mapping, x = self.patterns_wrapper[alpha]
                act_attack_string = attack_string_mapping[possib.attack_string]
                entity_to_attack_string[possib.candidate] = act_attack_string
            last_slots = max(
                len(attack_string.split(".")[0])
                for attack_string in entity_to_attack_string.values()
            )
            number_of_cases = 1 << last_slots
            if len(entry.alternatives) >= number_of_cases:
                self._data[epoch] = entry
                continue

            next_epoch_first_slot = (epoch + 1) * SLOTS_PER_EPOCH
            randao_reveals = [
                self.beaconchain[slot].randao_reveal
                for slot in range(
                    next_epoch_first_slot - last_slots, next_epoch_first_slot
                )
            ]

            if proposer_indices is None:
                proposer_indices = self.get_validators_fast(epoch=epoch + 2)

            original_distribution, _ = self.compute_proposer_indices(
                epoch=epoch + 2, indices=proposer_indices, additional_randao_reveals=[]
            )

            act_validators = self.actual_proposer_indices(epoch=epoch + 2)
            num_of_non_matching = sum(
                [a != b for a, b in zip(original_distribution, act_validators)]
            )
            if num_of_non_matching >= 2:
                print(
                    f"{BLUE}PREDICTION FROM ORIGINAL RANDOMNESS NOT MATCHING FOR EPOCH={epoch}{DEFAULT}"
                )
                print(
                    f"{BLUE}Predicted: {original_distribution}\nfactual: {self.actual_proposer_indices(epoch=epoch+2)}{DEFAULT}"
                )
                continue

            original_proposed_array = [
                self.beaconchain[slot].status == Status.PROPOSED
                for slot in range(
                    next_epoch_first_slot - SLOTS_PER_EPOCH, next_epoch_first_slot
                )
            ]

            for case in range(number_of_cases):
                binary_case = bin(case)[2:].zfill(last_slots)
                binary_array = [s == "1" for s in binary_case]

                if any(
                    changed and (randao_reveal is None or randao_reveal == MISSING)
                    for changed, randao_reveal in zip(binary_array, randao_reveals)
                ):
                    continue
                additional_randao_reveals = [
                    randao_reveal[2:]
                    for changed, randao_reveal in zip(binary_array, randao_reveals)
                    if changed
                ]
                if len(additional_randao_reveals) == 0:
                    indices = act_validators
                else:
                    indices, _ = self.compute_proposer_indices(
                        epoch=epoch + 2,
                        indices=proposer_indices,
                        additional_randao_reveals=additional_randao_reveals,
                    )
                key = list(original_proposed_array)

                for i, change in zip(
                    range(SLOTS_PER_EPOCH - last_slots, SLOTS_PER_EPOCH), binary_array
                ):
                    key[i] ^= change

                entry.alternatives[tuple(key)] = indices
            self._data[epoch] = entry


def produce_needed_alternatives(
    file_manager: FileManager, size_prefix: int, size_postfix: int
) -> None:
    preeval_mapping = preeval_grinding(
        size_prefix=size_prefix, size_postfix=size_postfix
    )
    assert preeval_mapping is not None
    patterns_wrapper = PatternsWrapper(mappings=preeval_mapping)
    data_to_attack_string = DataToAttackString(
        file_manager=file_manager, size_prefix=size_prefix, size_postfix=size_postfix
    )
    query_result = data_to_attack_string.query()

    alternatives_producer = AlternativesProducer(
        out_path="data/processed_data/alternatives.pkl",
        file_manager=file_manager,
        patterns_wrapper=patterns_wrapper,
    )
    alternatives_producer.start_grinding(attack_string_query=query_result)


def debug():
    serializer = PickleSerializer()
    with open("data/jsons/selfish_mixing/Kraken.json", "rb") as f:
        alternatives: dict[int, EpochAlternatives] = serializer.deserialize(f)

    print(alternatives[154280])


if __name__ == "__main__":
    debug()
