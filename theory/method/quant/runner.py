from typing import Optional
from theory.method.detailed_distribution import DetailedSlot
from theory.method.quant.base import QuantizedEAS, RANDAODataProvider


class QuantizedModelRunner:

    def __init__(
        self,
        eas_to_quantized_eas: dict[str, QuantizedEAS],
        mapping_by_eas_postf: dict[str, tuple[int, dict[str, int]]],
        eas_mapping: dict[str, str],
    ):
        for eas, quantized_eas in eas_to_quantized_eas.items():
            num, es_to_index = mapping_by_eas_postf[eas.split("#")[1]]
            quantized_eas.num_of_epoch_strings = num
            quantized_eas.epoch_string_to_index = es_to_index

        self.eas_to_quantized_eas = eas_to_quantized_eas
        self.mapping_by_eas_postf = mapping_by_eas_postf
        self.eas_mapping = eas_mapping

    def reset(self, eas: str) -> list[int]:
        """
        Resets to a class with a const eas

        Args:
            eas (str): extended attack string

        Returns:
            list[int]: list of configs that the attacker can calculate the RANDAO, this the (adv_slots, epoch_string).
            These should be the keys in the feed method.
        """
        self.captured_actions: list[DetailedSlot] = []
        self.config: int = 0
        self.head: int = 1
        self.eas = self.eas_mapping[eas]
        self.id_to_epoch: dict[int, tuple[int, str]] = {}
        self.before = True
        self.required = self.eas_to_quantized_eas[self.eas].run_prereq()

        self.prev_res: Optional[int] = None
        return self.required

    def consume(self, id_to_epoch: dict[int, tuple[int, str]]) -> tuple[int, bool, list[DetailedSlot]]:
        assert set(id_to_epoch) == set(
            self.required
        ), f"{set(id_to_epoch)=} {set(self.required)=}"
        assert all(
            id % self.head == self.config for id in id_to_epoch
        ), f"{self.head=} {self.config=} {id_to_epoch=}"
        # id_to_epoch = {self.config + self.head * key: val for key, val in id_to_epoch.items()}

        self.id_to_epoch.update(id_to_epoch)
        quantized_eas = self.eas_to_quantized_eas[self.eas]
        necessary_id_to_epoch = {
            id // self.head: epoch
            for id, epoch in self.id_to_epoch.items()
            if id % self.head == self.config
        }

        prev, rest = self.eas.split(".")
        if self.before:

            best_id, future_ids = quantized_eas.run(id_to_epoch=necessary_id_to_epoch)
            if quantized_eas.id_to_outcome[best_id].known:
                self.before = False
                future_ids = [id * self.head + self.config for id in future_ids]
                future_ids = [id for id in future_ids if id not in self.id_to_epoch]

                self.required = future_ids
                slot = -len(self.eas.split(".")[0])
                end_slot = quantized_eas.id_to_outcome[best_id].end_slot
                assert end_slot > slot
                actions = quantized_eas.id_to_outcome[best_id].det_slot_statuses
                actions = [action for action in actions if slot <= action.slot < end_slot]
                self.captured_actions.extend(
                    quantized_eas.id_to_outcome[best_id].det_slot_statuses
                )
                if len(future_ids) == 0:
                    assert quantized_eas.id_to_outcome[best_id].end_slot >= 0
                    assert (
                        0 <= best_id < 2 ** len(self.eas.split(".")[0])
                    ), f"{best_id=} {self.eas=}"
                    self.config += self.head * best_id
                    
                    slots = [action.slot for action in self.captured_actions]
                    assert len(slots) == len(
                        set(slots)
                    ), f"{self.eas=} {self.captured_actions=}"
                    self.prev_res = self.config
                    return self.config, False, actions

                self.prev_res = self.config + self.head * best_id
                return self.prev_res, True, actions
            else:
                slot = -len(self.eas.split(".")[0])
                end_slot = quantized_eas.id_to_outcome[best_id].end_slot
                assert end_slot > slot
                self.prev_res = self.config + self.head * best_id
                self.config += self.head * (best_id % (2 ** (end_slot - slot)))

                self.head *= 2 ** (end_slot - slot)
                actions = [
                    action for action in quantized_eas.id_to_outcome[best_id].det_slot_statuses
                    if slot <= action.slot < end_slot
                ]
                self.captured_actions.extend(actions)

                prev = prev[end_slot - slot :]
                self.eas = self.eas_mapping[f"{prev}.{rest}"]

                assert (self.eas in self.eas_to_quantized_eas) == (
                    prev != ""
                ), f"{self.eas=} {prev=}"

                if self.eas in self.eas_to_quantized_eas:
                    future_ids = self.eas_to_quantized_eas[self.eas].run_prereq()
                self.required = [self.config + self.head * id for id in future_ids]
                self.required = [
                    id for id in self.required if id not in self.id_to_epoch
                ]

                return self.prev_res, bool(self.required), actions
        else:
            assert (
                self.prev_res % self.head == self.config
            ), f"{self.prev_res=} {self.head=} {self.config=} {self.eas=}"
            best_id = quantized_eas.run_after(
                id_to_epoch=necessary_id_to_epoch,
                normalized_id=self.prev_res // self.head,
            )
            # if quantized_eas.id_to_outcome[best_id].known:
            self.prev_res = self.config + self.head * best_id
            slot = -len(self.eas.split(".")[0])
            end_slot = quantized_eas.id_to_outcome[best_id].end_slot
            assert end_slot > slot
            self.config += self.head * (best_id % (2 ** (end_slot - slot)))
            self.head *= 2 ** (end_slot - slot)
            actions = [
                action
                for action in quantized_eas.id_to_outcome[best_id].det_slot_statuses
                if slot <= action.slot < end_slot
            ]
            self.captured_actions.extend(actions)

            prev = prev[end_slot - slot :]
            self.eas = self.eas_mapping[f"{prev}.{rest}"]

            assert (self.eas in self.eas_to_quantized_eas) == (
                prev != ""
            ), f"{self.eas=} {prev=} {self.eas_to_quantized_eas=}"
            if self.eas in self.eas_to_quantized_eas:
                self.required = []
                self.before = True
                return self.prev_res, True, actions

            else:
                return self.prev_res, False, actions

    def run_one_epoch(self, eas: str, provider: RANDAODataProvider) -> int:
        """
        Runs the whole scenario getting epoch data from provider.

        Args:
            provider (RANDAODataProvider): consistent data provider

        Returns:
            int: config of chosen epoch info
        """
        self.reset(eas)

        while True:
            new_info: dict[int, tuple[int, str]] = {}
            for cfg in self.required:
                assert (
                    cfg not in provider.provided
                ), f"{eas=} {cfg=} {provider.provided=}"
                new_info[cfg] = provider.provide(cfg)
            best_cfg, more, actions = self.consume(id_to_epoch=new_info)
            if not more:
                break

            provider.feed_actions(actions=actions)
        provider.feed_result(best_cfg)
        return best_cfg
