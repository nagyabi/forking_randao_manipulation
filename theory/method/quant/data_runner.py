from base.beaconchain import EpochAlternatives
from base.statistics import CasesEntry
from theory.method.quant.base import RANDAODataProvider


class SimpleActualDataProvider(RANDAODataProvider):
    def __init__(self, entry: CasesEntry):
        super().__init__()
        self.len_before = len(entry.attack_string.split(".")[0])
        statuses_to_RO = dict(entry.statuses_to_RO)
        statuses_to_RO[entry.real_statuses] = entry.RO
        self.mapping = {
            self.__map(statuses): RO + statuses.count("m")
            for statuses, RO in statuses_to_RO.items()
        }

    def __map(self, statuses: str) -> int:
        res = 0
        statuses = statuses[: self.len_before]
        for status in statuses[::-1]:
            res *= 2
            if status == "p":
                res += 1
            else:
                assert status in "mr"

        return res

    def _provide(self, cfg: int) -> tuple[int, str]:
        if cfg not in self.mapping:
            raise KeyError(f"{cfg} not in {self.mapping}")
        result = self.mapping[cfg]
        return result, "#"


class AlternativesDataProvider(RANDAODataProvider):
    def __init__(
        self,
        entry: CasesEntry,
        epoch_alternatives: EpochAlternatives,
        orig_key: list[bool],
        index_to_entitiy: dict[int, str],
    ):
        super().__init__()
        self.entry = entry
        self.index_to_entity = index_to_entitiy
        self.len_before = len(entry.attack_string.split(".")[0])
        assert self.len_before > 0
        self.epoch_alternatives = epoch_alternatives
        self.orig_key = orig_key

    def _provide(self, cfg: int) -> tuple[int, str]:
        key = self.orig_key[: -self.len_before]
        for _ in range(self.len_before):
            key.append(cfg % 2 > 0)
            cfg //= 2
        indices = self.epoch_alternatives.alternatives[tuple(key)]
        return (
            sum(
                self.index_to_entity[index] == self.entry.attacker for index in indices
            ),
            "#",
        )
