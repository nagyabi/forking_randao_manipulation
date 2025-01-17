from dataclasses import dataclass
import json
import os
import pickle
import numpy as np

from base.beaconchain import EpochAlternatives
from base.helpers import FIGURES_FOLDER, LATEST_DELIVERY, SLOTS_PER_EPOCH, Status, sac
from base.statistics import read_delivery_cases, to_selfish_mixing
from data.file_manager import FileManager
from data.process.data_to_string import DataToAttackString
from make_figures.base import plot_data
from theory.method.cache import Cacher
from theory.method.preeval import calc_attack_string_mapping
from theory.method.quant.data_runner import AlternativesDataProvider
from theory.method.quant.runner import QuantizedModelRunner


def money_left_on_the_table(
    path: str, entities: list[str]
) -> dict[str, dict[int, tuple[int, int]]]:
    result: dict[int, tuple[int, int]] = {}

    rows = read_delivery_cases(path)
    rows = sorted(rows, key=lambda x: x.epoch)

    rows = to_selfish_mixing(rows)

    result: dict[str, dict[int, tuple[int, int, float]]] = {}
    for entity in entities:
        filtered = [row for row in rows if row.attacker == entity]
        assert len(filtered) > 0, f"{entity=}"
        result[entity] = {}

        cumRO = 0
        cumBEST = 0
        cumStake = 0.0

        for row in filtered:
            result[entity][row.epoch] = (cumRO, cumBEST, cumStake)

            cumRO += row.RO
            cumBEST += max(
                [row.RO, *[RO for RO in row.statuses_to_RO.values() if RO is not None]]
            )
            cumStake += row.stake * SLOTS_PER_EPOCH

    return result


@dataclass
class EmptyEpochInfo:
    epoch: int
    RO: int


def money_left_when_proposed(
    cases_path: str,
    entitiy: str,
    alpha: np.float64,
    size_prefix: int,
    size_postfix: int,
    iteration: int,
    min_epoch: int,
    max_epoch: int,
) -> dict[int, tuple[int, int]]:
    delivery_cases = read_delivery_cases(cases_path)
    beaconchain = FileManager.file_manager().beaconchain()
    index_to_entity = FileManager.file_manager().index_to_entity()
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
        alpha=alpha,
    )
    alternatives: dict[int, EpochAlternatives]
    with open("data/processed_data/alternatives.pkl", "rb") as f:
        alternatives = pickle.load(f)

    querier = DataToAttackString(
        file_manager=FileManager.file_manager(),
        size_prefix=size_prefix,
        size_postfix=size_postfix,
    )
    _, attack_string_mapping = calc_attack_string_mapping(
        alpha=alpha, size_prefix=size_prefix, size_postfix=size_postfix
    )

    print("All data read")
    query_res = querier.query()
    print("Query done")
    empty_epochs: list[int] = []
    for epoch, entry in query_res.items():
        if min_epoch <= epoch <= max_epoch:
            good_entries = [cand for cand in entry if cand.candidate == entitiy]
            assert 0 <= len(good_entries) <= 1
            if len(good_entries) == 0:
                empty_epochs.append(epoch)
            elif attack_string_mapping[good_entries[0].attack_string] == ".":
                empty_epochs.append(epoch)

    print(f"{len(empty_epochs)=}")

    filtered_delivery = [
        entry
        for entry in delivery_cases
        if entry.attacker == entitiy
        and entry.real_statuses.count("m") == 0
        and entry.real_statuses.count("r") == 0
        and min_epoch <= entry.epoch <= max_epoch
    ]
    filtered_epochs = set(entry.epoch for entry in filtered_delivery)
    exceptions = filtered_epochs.intersection(empty_epochs)

    # assert len(empty) == 0, f"{len(empty)=} {list(empty)[:10]=}"
    empty_epoch_infos: list[EmptyEpochInfo] = []
    for epoch in empty_epochs:
        RO = 0
        for slot in range((epoch + 2) * SLOTS_PER_EPOCH, (epoch + 3) * SLOTS_PER_EPOCH):
            if index_to_entity[beaconchain[slot].proposer_index] == entitiy:
                RO += 1
        empty_epoch_infos.append(EmptyEpochInfo(epoch=epoch, RO=RO))

    epochs = sorted([*empty_epoch_infos, *filtered_delivery], key=lambda x: x.epoch)
    result: dict[int, tuple[int, int, int, int]] = {}

    cumRO = 0
    cumSM = 0
    cumFORK = 0
    cumSLOTS = 0

    for entry in epochs:
        result[entry.epoch] = (cumRO, cumSM, cumFORK, cumSLOTS)

        cumRO += entry.RO
        if isinstance(entry, EmptyEpochInfo):
            cumSM += entry.RO
            cumFORK += entry.RO
            cumSLOTS += entry.RO
        else:
            entry.cut(size_postfix)
            eas = eas_mapping[f"{entry.attack_string}#"]
            if len(eas.split(".")[0]) > 0:
                provider = AlternativesDataProvider(
                    entry=entry,
                    epoch_alternatives=alternatives[entry.epoch],
                    orig_key=[
                        beaconchain[slot].status == Status.PROPOSED
                        for slot in range(
                            entry.epoch * SLOTS_PER_EPOCH,
                            entry.epoch * SLOTS_PER_EPOCH + SLOTS_PER_EPOCH,
                        )
                    ],
                    index_to_entitiy=index_to_entity,
                )
                cfg = runner.run_one_epoch(eas=eas, provider=provider)
                cumFORK += provider.provide(cfg)[0] - sac(
                    before=entry.attack_string.split(".")[0], cfg=cfg
                )
                cumSLOTS += provider.provide(cfg)[0]
                sm_ROs = [entry.RO]
                cutting = entry.attack_string.split(".")[0].rfind("h")
                cutting = cutting + 1 if cutting >= 0 else 0
                for statuses, RO in entry.statuses_to_RO.items():
                    if statuses[:cutting].count("p") == cutting:
                        sm_ROs.append(RO)
                cumSM += max(sm_ROs)
            else:
                cumSM += entry.RO
                cumFORK += entry.RO
                cumSLOTS += entry.RO

    return result


def plot_money_left_on_the_table(
    path: str, entities: list[str], cum_stake_for_entities: dict[str, bool]
) -> None:
    data = money_left_on_the_table(
        path=path,
        entities=entities,
    )
    base_folder = os.path.join(FIGURES_FOLDER, "money_on_the_table")
    os.makedirs(base_folder, exist_ok=True)

    for entity, graph in data.items():
        path = os.path.join(base_folder, f"{entity}.png")
        tkz_path = os.path.join(base_folder, f"{entity}.tex")

        plot_data(
            id_to_mapping={
                "real": {key: val[0] for key, val in graph.items()},
                "best": {key: val[1] for key, val in graph.items()},
                "cumStake": {key: val[2] for key, val in graph.items()},
            },
            id_to_name={
                "real": f"{entity} (actual)",
                "best": f"{entity} (potential)",
                "cumStake": f"{entity} (cumulative stake)",
            },
            id_to_color={"real": "blue", "best": "red", "cumStake": "brown"},
            title="Money left on the table",
            to_filename=path,
            tkz_filename=tkz_path,
            x_label="Epoch",
            left_y_label="Slots",
            right_y_label=None,
        )


def plot_money_left_when_proposed():
    result = money_left_when_proposed(
        cases_path=LATEST_DELIVERY,
        entitiy="Lido",
        alpha=np.float64(0.271),
        size_prefix=2,
        size_postfix=6,
        iteration=0,
        min_epoch=190000,
        max_epoch=314994,
    )
    with open("dumps/left_on_the_table2.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    # plot_money_left_when_proposed()
    # exit()

    with open("dumps/left_on_the_table2.json", "r") as f:
        data: dict[int, tuple[int, int, int]] = json.load(f)
    data = {int(key): val for key, val in data.items()}
    max_epoch = max(data)
    min_epoch = min(data)
    #
    print(f"Act: {data[max_epoch][0] - data[min_epoch][0]}")
    print(f"SM: {data[max_epoch][1] - data[min_epoch][1]}")
    print(f"Fork: {data[max_epoch][2] - data[min_epoch][2]}")
    print(f"Slots: {data[max_epoch][3] - data[min_epoch][3]}")
    # ma = []
    # for epoch, vals in data.items():
    #    if epoch+1 in data and data[epoch+1][1] - vals[1] > data[epoch+1][2] - vals[2]:
    #        ma.append(epoch)
    # print(len(ma))

    plot_data(
        id_to_mapping={
            "act": {key: val[0] for key, val in data.items() if key % 200 == 0},
            "sm": {key: val[1] for key, val in data.items() if key % 200 == 0},
            "fork": {key: val[2] for key, val in data.items() if key % 200 == 0},
            "slots": {key: val[3] for key, val in data.items() if key % 200 == 0},
        },
        id_to_name={
            "act": "Actual",
            "sm": "With selfish mixing",
            "fork": "With selfish mixing and forking",
            "slots": "All slots in epoch E+2",
        },
        id_to_color={"act": "blue", "sm": "brown", "fork": "red", "slots": "purple"},
        id_to_linestyle={
            "act": (0, (10, 5)),
            "sm": "-.",
            "fork": "-",
            "slots": ":",
        },
        title="Lido's rewards from unrealized RANDAO manipulations",
        to_filename=os.path.join(FIGURES_FOLDER, "lido_left_on_the_table.png"),
        tkz_filename=os.path.join(FIGURES_FOLDER, "lido_left_on_the_table.tex"),
        x_label="Epoch",
        right_y_label=None,
    )
