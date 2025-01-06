from helpers import minFork


def ExAnte(alpha: float, a_1: int, h: int, a_2: int, s: int) -> set[tuple[int, int]]:
    remain = 32 - s
    C: set[tuple[int, int]] = set()
    if a_1 < remain:
        C = {0}
        i = 1
        while a_1 + h + i <= remain and i < a_2:
            C.add(i)
            i += 1
    C_ = {(i, minFork(alpha, h + i)) for i in C}
    return {(i, b) for i, b in C_ if b <= a_1}
