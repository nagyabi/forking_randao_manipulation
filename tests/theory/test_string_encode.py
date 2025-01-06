import pytest

from theory.method.utils.attack_strings import EpochString, ExtendedAttackString
from theory.method.quant.quantize import StringEncoder


arguments = [
    (4, 2),
    (7, 1),
    (6, 3),
]


@pytest.mark.parametrize("size_prefix, size_postfix", arguments)
def test_string_encoding(size_prefix: int, size_postfix: int):
    encoder = StringEncoder(size_prefix=size_prefix, size_postfix=size_postfix)

    epoch_strings = [
        str(es)
        for es in EpochString.possibilities(
            size_pre=size_prefix, size_post=size_postfix
        )
    ]
    eass = [
        str(eas)
        for eas in ExtendedAttackString.possibilities(
            size_prefix=size_prefix, size_postfix=size_postfix
        )
    ]

    out_es = [encoder.encode_epoch_string(es) for es in epoch_strings]
    mapped_es = set(out_es)
    assert len(out_es) == len(mapped_es)
    assert mapped_es == set(range((size_prefix + 1) * 2**size_postfix))

    out_eass = [encoder.encode_eas(eas) for eas in eass]
    mapped_eas = set(out_eass)
    assert len(mapped_eas) == len(out_eass)
    assert mapped_eas == set(range((size_prefix + 1) * 2 ** (2 * size_postfix)))
