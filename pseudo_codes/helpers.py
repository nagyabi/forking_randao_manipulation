import math

PBOOST = 0.4


def minFork(alpha: float, h: int) -> int:
    return max(1, math.ceil((h * (1 - 2 * alpha) - PBOOST) / alpha))
