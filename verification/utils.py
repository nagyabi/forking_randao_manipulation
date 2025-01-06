from dataclasses import dataclass, is_dataclass
import hashlib
import struct
from py_ecc.bls import G2ProofOfPossession as bls_pop

GENESIS_VALIDATOR_ROOT = (
    "0x4b363db94e286120d76eb905340fdd4e54bfe9f06bf33ff6cf5ad27f511bfe95"
)

genesis_validators_root = bytes.fromhex(GENESIS_VALIDATOR_ROOT[2:])
DOMAIN_RANDAO = bytes.fromhex("02000000")
GENESIS_FORK_VERSION = bytes.fromhex("00000000")

BYTES_PER_CHUNK = 32


def next_pow_of_two(i):
    """Get the next power of 2 of i, with 0 mapping to 1."""
    if i == 0:
        return 1
    return 1 << (i - 1).bit_length()


def pad_bytes_to_chunks(data):
    """Pad the data to ensure it's aligned to BYTES_PER_CHUNK bytes."""
    padding_len = (BYTES_PER_CHUNK - len(data) % BYTES_PER_CHUNK) % BYTES_PER_CHUNK
    return data + (b"\x00" * padding_len)


def hash_data(data):
    """Compute the SHA-256 hash of the data."""
    return hashlib.sha256(data).digest()


def merkleize(chunks, limit=None):
    """Merkleize the chunks."""
    if limit is None:
        limit = next_pow_of_two(len(chunks))
    if limit < len(chunks):
        raise ValueError("Input exceeds the limit")

    chunks = chunks + [b"\x00" * BYTES_PER_CHUNK] * (
        next_pow_of_two(len(chunks)) - len(chunks)
    )

    def merkleize_helper(chunks):
        if len(chunks) == 1:
            return chunks[0]

        new_chunks = []
        for i in range(0, len(chunks), 2):
            left = chunks[i]
            right = chunks[i + 1] if i + 1 < len(chunks) else b"\x00" * BYTES_PER_CHUNK
            new_chunks.append(hash_data(left + right))

        return merkleize_helper(new_chunks)

    return merkleize_helper(chunks)


def pack_uint64(value):
    """Serialize a UInt64 value into bytes."""
    return struct.pack("<Q", value)  # Little-endian format


def merkleize_uint64(value):
    """Merkleize a UInt64 value."""
    serialized = pack_uint64(value)
    padded = pad_bytes_to_chunks(serialized)
    chunks = [
        padded[i : i + BYTES_PER_CHUNK] for i in range(0, len(padded), BYTES_PER_CHUNK)
    ]
    return merkleize(chunks)


def serialize_dataclass(instance):
    """Serialize a dataclass instance into bytes."""
    serialized = b""
    for field in instance.__dataclass_fields__.values():
        value = getattr(instance, field.name)
        assert isinstance(value, bytes)
        serialized += merkleize_bytes(value)
    return serialized


def merkleize_dataclass(instance):
    """Merkleize a dataclass instance."""
    serialized = serialize_dataclass(instance)
    padded = pad_bytes_to_chunks(serialized)
    chunks = [
        padded[i : i + BYTES_PER_CHUNK] for i in range(0, len(padded), BYTES_PER_CHUNK)
    ]
    return merkleize(chunks)


def merkleize_bytes(data):
    """Merkleize a bytes object."""
    padded = pad_bytes_to_chunks(data)
    chunks = [
        padded[i : i + BYTES_PER_CHUNK] for i in range(0, len(padded), BYTES_PER_CHUNK)
    ]
    return merkleize(chunks)


@dataclass
class SigningData:
    object_root: bytes
    domain: bytes


@dataclass
class ForkData:
    current_version: bytes
    genesis_validators_root: bytes


def hash_tree_root(value):
    if isinstance(value, bytes):
        return merkleize_bytes(value)
    elif isinstance(value, int):
        return merkleize_uint64(value=value)
    elif is_dataclass(value):
        return merkleize_dataclass(value)
    else:
        raise TypeError("Unsupported type for hash_tree_root")


def get_fork_version(epoch: int) -> bytes:
    forks: list[tuple[bytes, int]] = [
        (GENESIS_FORK_VERSION, 0),
        (bytes.fromhex("02000000"), 144896),
        (bytes.fromhex("03000000"), 194048),
        (bytes.fromhex("04000000"), 269568),
    ]
    for fork in reversed(forks):
        if epoch >= fork[1]:
            return fork[0]


def compute_domain(
    domain_type: bytes, fork_version: bytes, genesis_validators_root: bytes
) -> bytes:
    fork_data_root = compute_fork_data_root(fork_version, genesis_validators_root)
    return domain_type + fork_data_root[:28]


def get_domain(domain_type: bytes, epoch: int) -> bytes:
    fork_version = get_fork_version(epoch)
    return compute_domain(domain_type, fork_version, genesis_validators_root)


def compute_fork_data_root(
    current_version: bytes, genesis_validators_root: bytes
) -> bytes:
    return hash_tree_root(
        ForkData(
            current_version=current_version,
            genesis_validators_root=genesis_validators_root,
        )
    )


def compute_fork_digest(
    current_version: bytes, genesis_validators_root: bytes
) -> bytes:
    return compute_fork_data_root(current_version, genesis_validators_root)[:4]


def compute_signing_root(epoch: int, domain: bytes) -> bytes:
    return hash_tree_root(
        SigningData(
            object_root=hash_tree_root(epoch),
            domain=domain,
        )
    )


if __name__ == "__main__":
    epoch = 156411
    pub_key = bytes.fromhex(
        "b12c1fd2756304809bedddaf22f1bac2f4cf2589b389924d91ee45dced9a00a8aea3fc76539b6585a560bda341770aa1"
    )

    root = compute_signing_root(
        epoch=epoch, domain=get_domain(DOMAIN_RANDAO, epoch=epoch)
    )
    print(f"{root.hex()=}")

    sig = "0x956527c2d6c6e54f77114aba533aecae246b953ee927a003495d1c787b183dea6f1dbc2d4737536b7629ed1843fafbce017691d5697055bdd111f63765c942deefbd06c6cea079ae1011d6ab1c1f8aade160f307b6d26e092fc81520469a8e14"
    randao_reveal = bytes.fromhex(sig[2:])
    print(bls_pop.Verify(pub_key, root, randao_reveal))
