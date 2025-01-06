from collections import defaultdict
from base.helpers import BIG_ENTITIES, LATEST_DELIVERY, Comparison
from base.statistics import read_delivery_cases


def extract_statuses_from_delivery(
    delivery_path: str, entities: list[str]
) -> tuple[dict, dict]:
    delivery = read_delivery_cases(delivery_path)
    entity_to_alphas: dict[str, list[float]] = defaultdict(list)
    entity_to_comp: dict[str, list[int]] = defaultdict(lambda: [0] * 3)
    for row in delivery:
        if (
            row.attack_string.count("h") > 0
            or row.real_statuses.count("m") > 0
            or row.real_statuses.count("r") > 0
        ):
            continue
        if row.attacker not in entities:
            continue
        entity_to_alphas[row.attacker].append(row.stake)
        entity_to_comp[row.attacker][row.outcome.value - 1] += 1
    entity_to_alpha = {
        entity: sum(alphas) / len(alphas) for entity, alphas in entity_to_alphas.items()
    }
    return entity_to_alpha, entity_to_comp


def pretty_print_extracted(
    entities: list[str],
    entity_to_alpha: dict[str, float],
    entity_to_comp: dict[str, list[int]],
):
    for entity in entities:
        print(
            f"{entity}      &       {entity_to_comp[entity][Comparison.BEST.value-1]}  &     {entity_to_comp[entity][Comparison.NEUTRAL.value-1]} &    {entity_to_comp[entity][Comparison.WORSE.value-1]}  &    {round(100 * entity_to_alpha[entity], 2)}  \\\\"
        )


if __name__ == "__main__":
    entity_to_alpha, entity_to_comp = extract_statuses_from_delivery(
        LATEST_DELIVERY, BIG_ENTITIES
    )
    pretty_print_extracted(BIG_ENTITIES, entity_to_alpha, entity_to_comp)
