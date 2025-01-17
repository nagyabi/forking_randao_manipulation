import numpy as np
from theory.method.cache import Cacher
from theory.method.detailed_distribution import DetailedSlot
from theory.method.quant.base import RANDAODataProvider
from theory.method.quant.runner import QuantizedModelRunner


class UserRANDAOProvider(RANDAODataProvider):
    def __init__(self, eas: str):
        super().__init__()
        attack_string, _ = eas.split("#")
        self.before, self.after = attack_string.split(".")

    def prettyp_cfg(self, cfg: int):
        at_st_line = "|".join(self.before) + "||" + "|".join(self.after)
        print(at_st_line)
        print("=" * len(at_st_line))
        bin_cfg = bin(cfg)[2:]
        assert len(bin_cfg) <= len(self.before)
        bin_cfg = bin_cfg.zfill(len(self.before))[::-1]
        print("|".join(bin_cfg) + "||" + "|".join(" " * len(self.after)))

    def _provide(self, cfg) -> tuple[int, str]:
        self.prettyp_cfg(cfg)
        print()
        adv_slots = None
        epoch_string = None
        while True:
            try:
                adv_slots = int(input("adversarial slots: "))
                break
            except ValueError:
                pass
        while True:
            try:
                epoch_string = input("epoch string: ").strip().lower()
                assert epoch_string.count("#") == 1
                assert epoch_string.count("a") + epoch_string.count("h") + 1 == len(
                    epoch_string
                )
                break
            except AssertionError:
                pass
        return (adv_slots, epoch_string)

    def feed_result(self, cfg: int) -> None:
        print("(SUB)RES:")
        self.prettyp_cfg(cfg)
        print()
    
    def feed_actions(self, actions: list[DetailedSlot]):
        print(f"{actions=}")


def run_quant_model(
    alpha: np.float64, size_prefix: int, size_postfix: int, iteration: int
):
    cacher = Cacher(
        alpha=alpha, size_prefix=size_prefix, size_postfix=size_postfix, default=None
    )
    eas_mapping, eas_to_quant, mapping_by_eas_postf = cacher.get_quant(
        iteration=iteration
    )
    runner = QuantizedModelRunner(
        eas_to_quantized_eas=eas_to_quant,
        mapping_by_eas_postf=mapping_by_eas_postf,
        eas_mapping=eas_mapping,
    )
    while True:
        eas = input("eas: ")
        mapped = eas_mapping[eas]
        print(f"{eas} => {mapped}")
        runner.run_one_epoch(eas=eas, provider=UserRANDAOProvider(eas=eas))


if __name__ == "__main__":
    run_quant_model(
        alpha=np.float64(0.281),
        size_prefix=2,
        size_postfix=6,
        iteration=10,
    )
