import gc
from typing import Generic, Optional

import numpy as np
from tqdm import tqdm

from theory.method.utils.attack_strings import (
    EpochPostfix,
    EpochPrefix,
    EpochString,
    ExtendedAttackString,
)
from theory.method.detailed_distribution import (
    DetailedDistribution,
    T_Detailed_Distribution,
    max_detailed,
)
from theory.method.distribution import DetailedDistributionMaker, DistributionMaker
from theory.method.preeval import (
    calc_attack_string_mapping,
    calc_eas_mapping,
    calc_postfix_mapping,
)
from theory.method.quant.quantize import Quantizer
from theory.method.utils.voting import dynamic_vote


class DetailedEvaluator(Generic[T_Detailed_Distribution]):
    def __init__(
        self,
        maker: DistributionMaker[T_Detailed_Distribution],
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
        known: dict[str, bool],
        eas_mapping: dict[str, str],
        quantizer: Optional[Quantizer] = None,
        optimize: bool = False,
    ):
        self.maker = maker
        self.known = known
        self.eas_mapping = eas_mapping
        self.alpha = alpha
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix
        self.optimize = optimize

        self.eass: list[ExtendedAttackString] = ExtendedAttackString.possibilities(
            size_prefix=size_prefix, size_postfix=size_postfix
        )
        ordering = lambda x: (
            len(x.postfix_epoch_next.postfix),
            x.postfix_epoch_next.postfix,
            len(x.attack_string.postfix_prev.postfix),
            len(x.attack_string.prefix_next.prefix),
        )
        self.eass = sorted(self.eass, key=ordering)

        self.quantizer = quantizer

    def eval_eas(self, eas: ExtendedAttackString) -> T_Detailed_Distribution:
        eas_s = str(eas)

        if eas.attack_string.postfix_prev.postfix.count("h") == 0:
            n = eas.attack_string.postfix_prev.postfix.count("a")

            if n == 0:
                return self.null_cases[eas_s]

            assert eas_s[0] == "a"
            mapped_eas = self.eas_mapping[str(eas)]
            attack_string = str(eas.attack_string)
            assert self.known[attack_string], f"{mapped_eas=}"

            if str(eas) != mapped_eas:
                return self.eas_to_dist[mapped_eas]

            previous = self.eas_to_dist[eas_s[1:]]
            missed = previous.insert_noncanonical(is_adv_slot=True)
            proposed = previous.insert_canonical(is_adv_slot=True, hide=False)

            result = proposed.max(missed)

            if self.quantizer is not None:
                self.quantizer.quantize_sm(eas=eas, distribution=result)

            return result
        else:
            mapped_eas = self.eas_mapping[str(eas)]
            if str(eas) != mapped_eas:
                attack_string = str(eas.attack_string)
                if self.known[attack_string]:
                    return self.eas_to_dist[mapped_eas]
                else:
                    return self.eas_to_dist[mapped_eas].expected_value_in_distribution()

            extracted = (
                eas.attack_string.postfix_prev.postfix
                + eas.attack_string.prefix_next.prefix
            )
            assert extracted.count("h") > 0, f"{extracted=} {eas=}"

            adv_cluster1 = 0
            index = 0
            while extracted[index] == "a":
                index += 1
                adv_cluster1 += 1
            honest_cluster = 0
            while extracted[index] == "h":
                index += 1
                honest_cluster += 1
            adv_cluster2 = 0
            while (
                index < len(extracted)
                and extracted[index] == "a"
                and self.known.get(
                    f"{eas.attack_string.postfix_prev.postfix[index + 1:]}.{eas.attack_string.prefix_next}",
                    False,
                )
            ):
                index += 1
                adv_cluster2 += 1

            limit = len(eas.attack_string.postfix_prev.postfix) - (
                adv_cluster1 + honest_cluster + adv_cluster2
            )
            upper_addit, votes = dynamic_vote(
                alpha=self.alpha,
                adv_cluster1=adv_cluster1,
                honest_cluster=honest_cluster,
                adv_cluster2=adv_cluster2,
                epoch_boundary=limit,
            )

            addit_to_dist: dict[int, T_Detailed_Distribution] = {}

            for addit in range(upper_addit):
                jump = adv_cluster1 + honest_cluster + addit + 1
                before = eas.attack_string.postfix_prev.postfix[jump:]
                old_eas = (
                    f"{before}.{eas.attack_string.prefix_next}#{eas.postfix_epoch_next}"
                )
                known = True
                honest_addit = 0
                while old_eas[0] == "h":
                    old_eas = old_eas[1:]
                    known = False
                    honest_addit += 1

                if old_eas[0] == ".":

                    honest_addit += old_eas.split("#")[0].count("h")

                dist = self.eas_to_dist[old_eas].reset(regret=False)
                if not known:
                    dist = dist.expected_value_in_distribution()

                    for _ in range(honest_addit):
                        dist = dist.insert_canonical(is_adv_slot=False, hide=False)

                # Landing of voting
                if adv_cluster1 + honest_cluster + addit < len(
                    eas.attack_string.postfix_prev.postfix
                ):
                    dist = dist.insert_canonical(is_adv_slot=True, hide=False)

                # Missing addit blocks
                for _ in range(addit):
                    dist = dist.insert_noncanonical(is_adv_slot=True)

                # Forked honest blocks
                for _ in range(honest_cluster):
                    dist = dist.insert_noncanonical(is_adv_slot=False)

                addit_to_dist[addit] = dist

            before = eas.attack_string.postfix_prev.postfix[
                adv_cluster1 + honest_cluster :
            ]
            no_fork_string = f"{before}.{eas.attack_string.prefix_next.prefix}#{eas.postfix_epoch_next.postfix}"
            no_forking_dist = (
                self.eas_to_dist[no_fork_string]
                .expected_value_in_distribution()
                .reset(regret=False)
            )
            for _ in range(honest_cluster):
                no_forking_dist = no_forking_dist.insert_canonical(
                    is_adv_slot=False, hide=False
                )
            for _ in range(adv_cluster1):
                no_forking_dist = no_forking_dist.insert_canonical(
                    is_adv_slot=True, hide=False
                )
            if len(votes) == 0:
                return no_forking_dist

            regret_base = self.eas_to_dist[no_fork_string].reset(regret=True)
            for _ in range(honest_cluster):
                regret_base = regret_base.insert_canonical(
                    is_adv_slot=False, hide=False
                )

            present: list[T_Detailed_Distribution] = []
            future: list[T_Detailed_Distribution] = []

            for vote_group in votes:
                group: list[T_Detailed_Distribution] = []
                diff = (
                    adv_cluster1 - vote_group.free_slots_before - vote_group.regret_sac
                )
                assert 0 <= diff <= 1
                for plan in vote_group.plans:
                    dist = addit_to_dist[plan.addit]
                    for _ in range(plan.free_slots_after):
                        missed_head = dist.insert_noncanonical(is_adv_slot=True)
                        prop_head = dist.insert_canonical(is_adv_slot=True, hide=True)
                        dist = prop_head.max(missed_head)
                    middle: list[T_Detailed_Distribution] = []

                    for _ in range(plan.possibilities_middle):
                        middle = [
                            dist.insert_noncanonical(is_adv_slot=True)
                            for dist in middle
                        ]
                        middle.append(
                            dist.insert_canonical(is_adv_slot=True, hide=True)
                        )
                        dist = dist.insert_noncanonical(is_adv_slot=True)

                    dist = max_detailed(middle)

                    if diff == 1:
                        dist = dist.insert_canonical(is_adv_slot=True, hide=False)
                    group.append(dist)

                group_dist = max_detailed(group)
                regret = regret_base
                for _ in range(vote_group.regret_sac):
                    regret = regret.insert_noncanonical(is_adv_slot=True)

                if diff == 1:
                    regret = regret.insert_canonical(is_adv_slot=True, hide=False)

                future_group = [group_dist.max(regret)]
                for _ in range(vote_group.free_slots_before):
                    missed = [
                        dist.insert_noncanonical(is_adv_slot=True)
                        for dist in future_group
                    ]
                    proposed = [
                        dist.insert_canonical(is_adv_slot=True, hide=False)
                        for dist in future_group
                    ]
                    future_group = [*proposed, *missed]
                future.extend(future_group)

                group_from_present = group_dist.max_unknown(unknown=regret)
                for _ in range(vote_group.free_slots_before):
                    missed = group_from_present.insert_noncanonical(is_adv_slot=True)
                    proposed = group_from_present.insert_canonical(
                        is_adv_slot=True, hide=False
                    )
                    group_from_present = proposed.max(missed)
                present.append(group_from_present)

            fork_dist = max_detailed(present)

            result = fork_dist.max(no_forking_dist)

            if self.quantizer is not None:
                self.quantizer.quantize_fork(
                    eas=eas,
                    distribution=result,
                    regret_distributions=future,
                )

            return result

    def eval_all_eas(
        self, value_function: dict[str, np.float64], desc: str = ""
    ) -> dict[str, T_Detailed_Distribution]:
        self.eas_to_dist: dict[str, T_Detailed_Distribution] = {}
        self.null_cases: dict[str, T_Detailed_Distribution] = {}

        for prefix in EpochPrefix.possibilities(size=self.size_prefix):
            for postfix in EpochPostfix.possibilities(size=self.size_postfix):
                key = f".{prefix}#{postfix}"
                self.null_cases[key] = self.maker.make_detailed(
                    value_function=value_function,
                    postfix_next_epoch=postfix,
                    nullcase=f".{prefix}",
                )

        progress_bar = tqdm(total=len(self.eass), desc=desc)
        interval = len(
            EpochString.possibilities(
                size_pre=self.size_prefix, size_post=self.size_postfix
            )
        )
        pfix = None
        batch: set[str] = set()
        for i, eas in enumerate(self.eass):
            if self.optimize:
                if pfix != eas.postfix_epoch_next.postfix:
                    for st in batch:
                        self.eas_to_dist[st] = self.eas_to_dist[
                            st
                        ].expected_value_in_distribution()
                    gc.collect()  # Forcing garbage collection /each unique postfixes
                    batch = set()

                pfix = eas.postfix_epoch_next.postfix
                self.eas_to_dist[str(eas)] = self.eval_eas(eas=eas)
                batch.add(str(eas))
            else:
                self.eas_to_dist[str(eas)] = self.eval_eas(eas=eas)
            if (i + 1) % interval == 0:
                progress_bar.update(interval)
        progress_bar.close()
        return dict(self.eas_to_dist)


def main(
    alpha: np.float64,
    size_prefix: int,
    size_postfix: int,
) -> None:

    known, attack_string_mapping = calc_attack_string_mapping(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
    )
    eas_mapping = calc_eas_mapping(
        attack_string_mapping=attack_string_mapping,
        postfix_mapping=calc_postfix_mapping(attack_string_mapping),
    )

    quantizer = Quantizer(
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        eas_mapping=eas_mapping,
        quant_folder="dumps/quant/",
    )

    evaluator = DetailedEvaluator[DetailedDistribution](
        maker=DetailedDistributionMaker(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            eas_mapping=eas_mapping,
        ),
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        known=known,
        eas_mapping=eas_mapping,
        quantizer=quantizer,
    )

    value_function = {str(eas): np.float64(0.0) for eas in evaluator.eass}
    evaluator.eval_all_eas(value_function=value_function, desc="Detailed eval")


if __name__ == "__main__":
    main(
        alpha=np.float64(0.201),
        size_prefix=2,
        size_postfix=5,
    )
