import pytest

from verification.utils import compute_domain, compute_signing_root


arguments_domain = [
    (
        "02000000",
        "02000000",
        "4b363db94e286120d76eb905340fdd4e54bfe9f06bf33ff6cf5ad27f511bfe95",
        "020000004a26c58b08add8089b75caa540848881a8d4f0af0be83417a85c0f45",
    ),
    (
        "120c0b0a",
        "8200fe04",
        "0c821d72182df0fcc5321fda64d3cf93aeca6ae2ae669cc4e292d169465c6b3b",
        "120c0b0a0469da93e9e8265881201816dfd88d1c7eb096c3959a1e693abee1d3",
    ),
    (
        "06b2e720",
        "27bb716a",
        "452d8e6f4c131916b5806f321175696eab89fc0f28ed85954ca913f8d6a91c24",
        "06b2e720bf6e82b316107306f92cc798be181a9ea630f375128c208081c559ce",
    ),
    (
        "9f8cafe0",
        "1c84396f",
        "8e601982bc156dd6f1a92552dda6730bfb5b8fb8577c00a55536d5da4afe3331",
        "9f8cafe096ee0fe82e67001b30ff1aac1ae6d8a495d7cf246b40aabbe9745426",
    ),
]


@pytest.mark.parametrize(
    "domain, fork_version, gen_val_root, expected", arguments_domain
)
def test_compute_domain(
    domain: str, fork_version: str, gen_val_root: str, expected: str
):
    exp_byt = compute_domain(
        bytes.fromhex(domain), bytes.fromhex(fork_version), bytes.fromhex(gen_val_root)
    )
    assert exp_byt.hex() == expected, f"{len(exp_byt.hex())=} {len(expected)=}"


arguments_root = [
    (
        1567,
        "739e8220328adda47ce6b3bafaf1441de47452de930b4ec753414a1046893d0b",
        "fd0c38f7359d7a85a439598c57b8f35a3d0cd347a0a2e405e45fcf5055fae1b1",
    ),
    (
        55565,
        "ea8e1441885f095bf38d3941f5392b38fed0a1c2b8e1a944e0fb22ab176a0876",
        "7e723fda5a3ae42fbb89cecc1f879323f26e37253cb114a854b022b13ef6b023",
    ),
    (
        11234,
        "3a056cf897a04aea57fd10d0dd1398e0ca242503b968130d9b4df69bf1b664a8",
        "4a010341007550dd9344e496ddb9b8eae546a504e96a4bf011a842573cd5e2e3",
    ),
]


@pytest.mark.parametrize("epoch, domain, expected", arguments_root)
def test_compute_signing_root(epoch: int, domain: str, expected: str):
    root_byt = compute_signing_root(epoch, bytes.fromhex(domain))
    assert root_byt.hex() == expected
