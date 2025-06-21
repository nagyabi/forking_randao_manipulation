import numpy as np
from base.helpers import SLOTS_PER_EPOCH
from theory.method.cache import Cacher
from theory.method.detailed_distribution import DetailedSlot
from theory.method.engine import BeaconChainAction, Propose, Vote
from theory.method.quant.base import RANDAODataProvider
from theory.method.quant.runner import QuantizedModelRunner


class UserRANDAOProvider(RANDAODataProvider):
    def __init__(self, eas: str, size_prefix: int, size_postfix: int):
        super().__init__()
        attack_string, _ = eas.split("#")
        self.before, self.after = attack_string.split(".")
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix

    def prettyp_cfg(self, cfg: int):
        at_st_line = "|".join(self.before) + "||" + "|".join(self.after)
        print(at_st_line)
        print("=" * len(at_st_line))
        bin_cfg = bin(cfg)[2:]
        assert len(bin_cfg) <= len(self.before)
        bin_cfg = bin_cfg.zfill(len(self.before))[::-1]
        print("|".join(bin_cfg) + "||" + "|".join(" " * len(self.after)))

    def _provide(self, cfg) -> tuple[int, str]:
        print("Provide the values to the RANDAO of")
        self.prettyp_cfg(cfg)
        print()
        adv_slots = None
        epoch_string = None
        while True:
            try:
                adv_slots = int(input("Adversarial slots: "))
                break
            except ValueError:
                pass
        while True:
            try:
                epoch_string = input("Epoch string: ").strip().lower()
                assert epoch_string.count("a") + epoch_string.count("h") + 1 == len(
                    epoch_string
                )
                assert epoch_string.count("#") <= 1
                if epoch_string.count("#") == 1:
                    before, after = epoch_string.split("#")
                    before = before[: self.size_prefix]
                    after = after[-self.size_postfix :]
                else:
                    assert len(epoch_string) == SLOTS_PER_EPOCH
                    before = epoch_string[: self.size_prefix]
                    after = epoch_string[-self.size_postfix :]
                if before.count("a") == 0:
                    before = ""
                else:
                    before = before[: before.find("a") + 1]
                epoch_string = f"{before}#{after}"

                break
            except AssertionError:
                pass
        return (adv_slots, epoch_string)

    def feed_result(self, cfg: int) -> None:
        print("Resulting RANDAO:")
        self.prettyp_cfg(cfg)
        print()
        print("=================")
        print()
        print()

    def feed_actions(self, actions: list[BeaconChainAction]) -> None:
        def slot_and_epoch(slot: int) -> str:
            epoch = "e" if slot < 0 else "e+1"
            slot %= SLOTS_PER_EPOCH
            return f"epoch: {epoch}; slot: {slot}"

        print()
        for action in actions:
            act_str: str
            if isinstance(action, Propose):
                act_str = f"PROPOSE Block ON <{slot_and_epoch(action.block_slot)}> PARENT: <{slot_and_epoch(action.parent_block_slot)}>"
            elif isinstance(action, Vote):
                act_str = f"VOTE for <{slot_and_epoch(action.target_slot)}>"
            print(f"[{slot_and_epoch(action.slot)}]", act_str, sep="\t")
        print()


def run_quant_model(
    alpha: np.float64, size_prefix: int, size_postfix: int, iteration: int
):
    cacher = Cacher(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        default=None,
        should_exists=True,
    )
    eas_mapping, eas_to_quant, mapping_by_eas_postf = cacher.get_quant(
        iteration=iteration
    )
    print()
    print("Model loaded successfully")
    print()
    runner = QuantizedModelRunner(
        eas_to_quantized_eas=eas_to_quant,
        mapping_by_eas_postf=mapping_by_eas_postf,
        eas_mapping=eas_mapping,
        alpha=alpha,
    )
    print(
        'Please provide an extended attack string (eas, e.g. "aha.a#aa") excluding ones starting with "." as they do not provide a manipulating opportunity.'
    )
    print()
    while True:
        eas = input("eas: ")
        mapped = eas_mapping[eas]
        print(f"{eas} => {mapped}")
        provider = UserRANDAOProvider(
            eas=eas, size_prefix=size_prefix, size_postfix=size_postfix
        )
        runner.run_one_epoch(eas=eas, provider=provider)


if __name__ == "__main__":
    run_quant_model(
        alpha=np.float64(0.281),
        size_prefix=2,
        size_postfix=6,
        iteration=10,
    )
