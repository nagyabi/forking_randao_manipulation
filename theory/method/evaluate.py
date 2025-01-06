import gc
from typing import Generic
import numpy as np
from tqdm import tqdm

from theory.method.utils.attack_strings import (
    AttackString,
    EpochPostfix,
    ExtendedAttackString,
)
from theory.method.distribution import (
    DistMakerBase,
    DistributionMaxer,
    T_Distribution,
    ValuedDistribution,
    ValuedDistributionMaker,
    selfish_mixing,
)
from theory.method.preeval import (
    calc_attack_string_mapping,
    calc_eas_mapping,
    calc_postfix_mapping,
)
from theory.method.utils.voting import dynamic_vote


class Evaluator(Generic[T_Distribution]):
    def __init__(
        self,
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
        known: dict[str, bool],
        eas_mapping: dict[str, str],
        optimize: bool = True,
    ) -> None:
        self.known = known
        self.eas_mapping = eas_mapping
        self.alpha = alpha
        self.size_prefix = size_prefix
        self.size_postfix = size_postfix
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
        self.optimize = optimize

    def eval_eas(self, eas: ExtendedAttackString) -> T_Distribution:
        prev = str(eas.attack_string.postfix_prev)
        if prev.count("h") == 0:
            t = prev.count("a")
            return self.t_slots_and_last_string_to_dist[
                (t, str(eas.postfix_epoch_next))
            ]
        elif prev[-1] == "h" and eas.attack_string.prefix_next.prefix == "":
            return self.t_slots_and_last_string_to_dist[
                (0, str(eas.postfix_epoch_next))
            ].expected_value_in_distribution()
        assert prev[0] == "a", f"{prev=}"

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
        while index < len(extracted) and extracted[index] == "a":
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
        addit_to_distribution: dict[int, T_Distribution] = {}

        for addit in range(upper_addit):
            jump = adv_cluster1 + honest_cluster + addit + 1
            before = prev[jump:]
            old_eas = (
                f"{before}.{eas.attack_string.prefix_next}#{eas.postfix_epoch_next}"
            )
            known = True
            while old_eas[0] == "h":
                old_eas = old_eas[1:]
                known = False

            dist = self.eas_to_dist[old_eas]
            if not known:
                dist = dist.expected_value_in_distribution()
            addit_to_distribution[addit] = dist.increase_sacrifice(addit)

        no_fork_string = (
            prev[adv_cluster1 + honest_cluster :]
            + "."
            + eas.attack_string.prefix_next.prefix
            + "#"
            + eas.postfix_epoch_next.postfix
        )
        no_forking_dist = self.eas_to_dist[
            no_fork_string
        ].expected_value_in_distribution()
        if len(votes) == 0:
            return no_forking_dist
        regret_distribution = self.eas_to_dist[
            no_fork_string
        ]  # The attacker can sometimes "regret" for trying to fork out the honest blocks
        if hasattr(regret_distribution, "regret"):
            regret_distribution = regret_distribution.regret()

        known_distributions: list[T_Distribution] = []

        for vote_group in votes:
            regret = regret_distribution.increase_sacrifice(
                sacrifice=vote_group.regret_sac
            )
            dists: list[T_Distribution] = []
            for plan in vote_group.plans:
                dist = selfish_mixing(
                    addit_to_distribution[plan.addit], slots=plan.free_slots_after
                )
                maxer = DistributionMaxer(
                    base_distribution=dist.increase_sacrifice(plan.base_sacrifice)
                )
                dist = maxer.max(number_of_dist=plan.possibilities_middle)
                dists.append(dist)
            dist = dists[0]
            for d in dists[1:]:
                dist = dist.max(d)
            if hasattr(dist, "increase_forked_honest_blocks"):
                dist = dist.increase_forked_honest_blocks(
                    forked_honest_blocks=honest_cluster
                )
            dist = dist.max_unknown(regret)
            dist = selfish_mixing(dist, slots=vote_group.free_slots_before)
            known_distributions.append(dist)

        fork_distribution = known_distributions[0]
        for dist in known_distributions[1:]:
            fork_distribution = fork_distribution.max(dist)
        distribution = fork_distribution.max(no_forking_dist)
        return distribution

    def eval_all_eas(
        self,
        value_function: dict[str, np.float64],
        maker: DistMakerBase[T_Distribution],
        desc: str = "",
    ) -> dict[str, T_Distribution]:
        self.eas_to_dist: dict[str, T_Distribution] = {}
        self.t_slots_and_last_string_to_dist: dict[tuple[int, str], T_Distribution] = {}
        for last_str in EpochPostfix.possibilities(size=self.size_postfix):
            dist = maker.make_distribution(
                value_function=value_function, postfix_next_epoch=last_str
            )

            for t in range(self.size_postfix + 1):
                self.t_slots_and_last_string_to_dist[(t, str(last_str))] = dist
                if t != self.size_postfix:
                    dist = dist.max(dist.increase_sacrifice(sacrifice=1))

        progress_bar = tqdm(total=len(self.eass), desc=desc)
        interval = len(
            AttackString.possibilities(
                size_postf_prev=self.size_postfix, size_pref_next=self.size_prefix
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


if __name__ == "__main__":
    alpha = np.float64(0.201)
    size_prefix = 2
    size_postfix = 6

    known, attack_string_mapping = calc_attack_string_mapping(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
    )
    eas_mapping = calc_eas_mapping(
        attack_string_mapping=attack_string_mapping,
        postfix_mapping=calc_postfix_mapping(attack_string_mapping),
    )

    maker = ValuedDistributionMaker(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        eas_mapping=eas_mapping,
    )
    evaluator = Evaluator[ValuedDistribution](
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        known=known,
        eas_mapping=eas_mapping,
    )
    dists = evaluator.eval_all_eas(
        value_function={str(eas): np.float64(0.0) for eas in evaluator.eass},
        maker=maker,
    )
    while True:
        eas = input(">>>> ")
        if eas not in dists:
            print(f"{eas} not in res")
        else:
            print(dists[eas].expected_value())
