import pickle
from base.helpers import SLOTS_PER_EPOCH, Status
from base.statistics import get_index_to_entity
from data.collect.beaconchain_grinder import BeaconChainGrinder
from data.collect.validator_grinder import (
    AddressGrinder,
    DepositGrinder,
    ValidatorGrinder,
)
from data.file_manager import FileManager
from data.internet.beaconcha.api import BeaconChaAPIConnection


def data_full(file_manager: FileManager, regenerate_validators: bool):
    beaconcha_connection = BeaconChaAPIConnection()
    assert (
        beaconcha_connection.is_available()
    ), f"[beaconchain] API connection not available"
    latest_epoch = beaconcha_connection.latest_epoch()
    # Cropping down precision so when running the function close to each other max_epoch will be the same
    max_epoch = (latest_epoch - 100) // 100 * 100

    beaconchain_grinder = BeaconChainGrinder(
        connection=beaconcha_connection, out_path="data/jsons/beaconchain.json"
    )
    beaconchain = beaconchain_grinder.start_grinding(max_epoch=max_epoch)
    assert beaconchain is not None
    max_slot = (max_epoch + 1) * SLOTS_PER_EPOCH - 1
    assert max_slot in beaconchain
    validator_grinder = ValidatorGrinder(
        out_path="data/jsons/validators.json",
        regenerate=regenerate_validators,
        length=100,
    )
    validators = validator_grinder.start_grinding()
    assert validators is not None
    slot = max_slot
    while beaconchain[slot].status != Status.PROPOSED:
        slot -= 1
    max_block = beaconchain[slot].block
    deposit_grinder = DepositGrinder(
        out_path="data/jsons/deposits.json", top_block=max_block
    )
    deposits = deposit_grinder.start_grinding()
    assert deposits is not None
    txns = [deposit.txn_hash for deposit in deposits]
    address_grinder = AddressGrinder(out_path="data/jsons/address_txn.json")
    address_txns = address_grinder.start_grinding(txns=txns)
    assert address_txns is not None
    file_manager._beaconchain.data = beaconchain
    file_manager._validators.data = validators
    file_manager._deposits.data = deposits
    file_manager._address_txn.data = address_txns
    index_to_entity = get_index_to_entity(file_manager)
    with open("data/processed_data/index_to_entity.pkl", "wb") as f:
        pickle.dump(index_to_entity, f)
    print("Data full collection succesful!")
