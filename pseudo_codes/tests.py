from ex_ante import ExAnte
import pytest

from fork_action import ForkAction

exante_arguments = [
    (0.2001, 1, 1, 1, 10, {(0, 1)}),
    (0.2001, 1, 1, 1, 29, {(0, 1)}),
    (0.2001, 1, 1, 1, 30, {(0, 1)}),
    (0.2001, 1, 1, 1, 31, set()),
    (0.199, 1, 1, 1, 10, set()),
    (0.199, 1, 1, 1, 29, set()),
    (0.199, 1, 1, 1, 30, set()),
    (0.199, 1, 1, 1, 31, set()),
    (0.2001, 1, 1, 2, 28, {(0, 1)}),
    (0.2001, 2, 1, 1, 28, {(0, 1)}),
    (0.1501, 2, 1, 1, 28, {(0, 2)}),
    (0.1491, 2, 1, 1, 28, set()),
    (0.3201, 1, 1, 2, 28, {(0, 1), (1, 1)}),
    (0.3201, 1, 1, 2, 29, {(0, 1), (1, 1)}),
    (0.3201, 1, 1, 2, 30, {(0, 1)}),
    (0.3199, 1, 1, 2, 28, {(0, 1)}),
]


@pytest.mark.parametrize("alpha, a_1, h, a_2, s, expected", exante_arguments)
def test_ex_ante(
    alpha: float, a_1: int, h: int, a_2: int, s: int, expected: set[tuple[int]]
):
    assert ExAnte(alpha, a_1, h, a_2, s) == expected


fork_arguments = [
    ("CCN", 3, 1, "PHM"),
    ("CCC", 3, 1, "PPH"),
    ("CNN", 3, 1, "HMM"),
    ("NNN", 3, 1, None),
    ("NCC", 3, 2, "MHH"),
    ("NCC", 3, 3, None),
]


@pytest.mark.parametrize("plan, a_1, n, expected", fork_arguments)
def test_forkaction(plan: str, a_1: int, n: int, expected: str | None):
    assert ForkAction(plan, a_1, n) == expected
