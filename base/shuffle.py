import numpy as np

from base.helpers import SLOTS_PER_EPOCH, bytes_to_uint64, sha_256, uint_to_bytes

SHUFFLE_ROUND_COUNT = 90
DOMAIN_BEACON_PROPOSER = bytes.fromhex("00000000")


def compute_shuffled_index(
    index: np.uint64, index_count: np.uint64, seed: bytes
) -> np.uint64:
    """
    Return the shuffled index corresponding to ``seed`` (and ``index_count``).
    """
    assert index < index_count

    # Swap or not (https://link.springer.com/content/pdf/10.1007%2F978-3-642-32009-5_1.pdf)
    # See the 'generalized domain' algorithm on page 3
    for current_round in range(SHUFFLE_ROUND_COUNT):
        pivot = (
            bytes_to_uint64(sha_256(seed + uint_to_bytes(np.uint8(current_round)))[0:8])
            % index_count
        )
        flip = (pivot + index_count - index) % index_count
        position = max(index, flip)
        source = sha_256(
            seed
            + uint_to_bytes(np.uint8(current_round))
            + uint_to_bytes(np.uint32(position // 256))
        )
        byte = np.uint8(source[int((position % 256) // 8)])
        bit = (int(byte) >> int((position % 8))) % 2
        index = flip if bit else index

    return int(index)


def dummy_compute_proposer_index(indices: list[int], seed: bytes) -> int:
    # MAX_RANDOM_BYTE = 2**8 - 1
    i = np.uint64(0)
    total = np.uint64(len(indices))

    candidate_index = indices[compute_shuffled_index(i % total, total, seed)]
    return candidate_index


def get_seed(mix: bytes, epoch: int, domain_type: bytes) -> bytes:
    """
    Return the seed at ``epoch``.
    """
    return sha_256(domain_type + uint_to_bytes(np.uint64(epoch)) + mix)


def get_beacon_proposer_index(
    mix: bytes, indices: list[int], epoch: int, slot: int
) -> int:
    """
    Return the beacon proposer index at the current slot.
    """
    seed = sha_256(
        get_seed(mix, epoch, DOMAIN_BEACON_PROPOSER) + uint_to_bytes(np.uint64(slot))
    )
    return dummy_compute_proposer_index(indices, seed)


def generate_epoch(
    mix: bytes, indices: list[int], epoch: int, from_slot: int
) -> list[int]:
    return [
        get_beacon_proposer_index(mix=mix, indices=indices, epoch=epoch, slot=slot)
        for slot in range(from_slot, from_slot + SLOTS_PER_EPOCH)
    ]


def mix_randao(randomness: bytes, randao_reveal: str) -> bytes:
    randao_bytes = bytes.fromhex(randao_reveal)
    randao_hashed = sha_256(randao_bytes)
    randao_contrib_int = int.from_bytes(randao_hashed, byteorder="big")
    randomness_int = int.from_bytes(randomness, byteorder="big")
    newrandomness_int = randao_contrib_int ^ randomness_int
    return newrandomness_int.to_bytes(32, byteorder="big")
