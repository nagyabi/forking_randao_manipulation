from tqdm import tqdm
from base.grinder import Grinder
from base.helpers import SLOTS_PER_EPOCH, Status
from data.file_manager import FileManager
from data.process.serialize import IntDictSerializer
from verification.utils import (
    DOMAIN_RANDAO,
    compute_domain,
    compute_signing_root,
    genesis_validators_root,
    get_domain,
)
from py_ecc.bls import G2ProofOfPossession as bls_pop

VERSION_LIMIT = 20


class BLSVerifier:
    def __init__(self, file_manager: FileManager):
        self.beaconchain = file_manager.beaconchain()
        self.validators = file_manager.validators()

    def verify_from_fork_version(self, fork_version: bytes, slot: int) -> bool:
        epoch = slot // SLOTS_PER_EPOCH
        domain = compute_domain(DOMAIN_RANDAO, fork_version, genesis_validators_root)
        root = compute_signing_root(epoch=epoch, domain=domain)
        randao_reveal = bytes.fromhex(self.beaconchain[slot].randao_reveal[2:])
        public_key = bytes.fromhex(
            self.validators[self.beaconchain[slot].proposer_index].public_key[2:]
        )
        return bls_pop.Verify(public_key, root, randao_reveal)

    def verify(self, slot: int) -> bool:
        epoch = slot // SLOTS_PER_EPOCH
        domain = get_domain(domain_type=DOMAIN_RANDAO, epoch=epoch)
        root = compute_signing_root(epoch=epoch, domain=domain)
        randao_reveal = bytes.fromhex(self.beaconchain[slot].randao_reveal[2:])
        public_key = bytes.fromhex(
            self.validators[self.beaconchain[slot].proposer_index].public_key[2:]
        )
        return bls_pop.Verify(public_key, root, randao_reveal)


def get_versions() -> dict[int, str]:
    verifier = BLSVerifier(FileManager.file_manager())
    min_epoch = min(verifier.beaconchain) // SLOTS_PER_EPOCH
    max_epoch = max(verifier.beaconchain) // SLOTS_PER_EPOCH

    def to_version(num: int) -> str:
        return f"{str(num).zfill(2)}000000"

    def find_version_to_slot(slot: int) -> int:
        for i in range(VERSION_LIMIT + 1):
            if verifier.verify_from_fork_version(
                bytes.fromhex(to_version(i)), slot=slot
            ):
                return i
        raise ValueError(f"Could't match any version to {slot=}")

    def find_version_to_epoch(epoch: int) -> int:
        for slot in range(epoch * SLOTS_PER_EPOCH, (epoch + 1) * SLOTS_PER_EPOCH):
            if verifier.beaconchain[slot].status == Status.PROPOSED:
                return find_version_to_slot(slot)
        raise ValueError(f"No proposed slot found in {epoch=}")

    def good(epoch_to_version: list[tuple[int, int]]) -> bool:
        for (prev_epoch, prev_version), (epoch, version) in zip(
            epoch_to_version[:-1], epoch_to_version[1:]
        ):
            if prev_version != version and prev_epoch + 1 != epoch:
                return False
        return True

    epoch_to_version: list[tuple[int, int]] = [
        (min_epoch, find_version_to_epoch(min_epoch)),
        (max_epoch, find_version_to_epoch(max_epoch)),
    ]
    while not good(epoch_to_version):
        new_epoch_to_version: list[tuple[int, int]] = []
        for (prev_epoch, prev_version), (epoch, version) in zip(
            epoch_to_version[:-1], epoch_to_version[1:]
        ):
            new_epoch_to_version.append((prev_epoch, prev_version))
            if prev_version != version and prev_epoch + 1 != epoch:
                assert prev_epoch < epoch
                assert prev_version < version
                new_epoch = (prev_epoch + epoch) // 2
                new_version = find_version_to_epoch(new_epoch)
                assert prev_version <= new_version <= version
                new_epoch_to_version.append((new_epoch, new_version))
                print(f"For epoch={new_epoch} version is {new_version}")

        new_epoch_to_version.append(epoch_to_version[-1])
        epoch_to_version = new_epoch_to_version
    result: dict[int, str] = {
        epoch_to_version[0][0]: to_version(epoch_to_version[0][1])
    }
    for (prev_epoch, prev_version), (epoch, version) in zip(
        epoch_to_version[:-1], epoch_to_version[1:]
    ):
        if prev_version != version:
            result[epoch] = to_version(version)
    return result


class ReorgVerifier(Grinder[dict[int, bool]]):
    """
    Verifies all the RANDAO reveals of the reorged slots
    """

    def __init__(self, file_manager: FileManager, out_path: str):
        super().__init__(
            out_path=out_path,
            default_data={},
            serializer=IntDictSerializer(),
            regenerate=False,
            safety_save=False,
        )
        self.verifier = BLSVerifier(file_manager=file_manager)
        self.beaconchain = file_manager.beaconchain()

    def _grind(self, **kwargs):
        reorged_slots = [
            slot
            for slot, block in self.beaconchain.items()
            if block.status == Status.REORGED
        ]
        reorged_slots = [slot for slot in reorged_slots if slot not in self._data]

        for slot in tqdm(reorged_slots, desc="Verfying"):
            self._data[slot] = self.verifier.verify(slot)
            self._data_changed = True
            assert self._data[slot], f"RANDAO reveal not matching for {slot=}"


if __name__ == "__main__":
    reorg_grinder = ReorgVerifier(
        FileManager.file_manager(), out_path="data/processed_data/bls_verified.json"
    )
    reorg_grinder.start_grinding()
