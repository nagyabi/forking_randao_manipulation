from collections import defaultdict
import json
import os

from matplotlib import pyplot as plt
from base.helpers import CACHE_FOLDER, FIGURES_FOLDER
from make_figures.base import STK_X_LABEL, cache_mapping, plot_data
from theory.method.preeval import calc_attack_string_mapping

from tikzplotlib import save as tikz_save


def plot_expected_values_by_as_groups(size_prefix: int, size_postfix: int) -> None:
    mapping = cache_mapping(
        base_folder=CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    honest_mapping: dict[float, float] = {}
    sm_mapping: dict[float, float] = {}
    fork_mapping: dict[float, float] = {}
    for alpha, folder in mapping.items():
        with open(os.path.join(folder, "attack_strings_to_probability.json"), "r") as f:
            local_mapping: dict = json.load(f)
        honest_mapping[alpha] = local_mapping["."]["RO"]["scaled"]
        sm_mapping[alpha] = sum(
            local_mapping["a" * n + "."]["RO"]["scaled"]
            for n in range(1, size_postfix + 1)
        )
        fork_mapping[alpha] = sum(
            entry["RO"]["scaled"]
            for attack_string, entry in local_mapping.items()
            if attack_string.count("h") > 0
        )

    path = os.path.join(FIGURES_FOLDER, "expected_values_by_attack_string.png")
    tkz_path = os.path.join(FIGURES_FOLDER, "expected_values_by_attack_string.tex")
    plot_data(
        id_to_mapping={
            "honest": honest_mapping,
            "sm": sm_mapping,
            "fork": fork_mapping,
        },
        # order=["honest", "sm", "fork"],
        id_to_color={
            "honest": "blue",
            "sm": "orange",
            "fork": "green",
        },
        id_to_name={
            "honest": "Empty attack string",
            "sm": "Attack strings for selfish mixing",
            "fork": "Attack strings with forking possibility",
        },
        id_to_linestyle={
            "honest": (0, (10, 5)),
            "sm": "-.",
            "fork": "--",
        },
        title="Expected values from attack strings",
        x_label=STK_X_LABEL,
        left_y_label=r"Effective stakes ($\alpha$)",
        right_y_label=None,
        to_filename=path,
        tkz_filename=tkz_path,
    )


def plot_chaos(
    size_prefix: int,
    size_postfix: int,
    colors: list[str],
    capture: int,
    before_limit: int,
):
    mapping = cache_mapping(
        base_folder=CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    data = defaultdict(dict)
    max_alpha, max_folder = max(mapping.items(), key=lambda x: x[0])
    with open(os.path.join(max_folder, "attack_strings_to_probability.json"), "r") as f:
        max_mapping: dict = json.load(f)

    attack_strings: list[str] = list(max_mapping)

    for alpha, folder in mapping.items():
        with open(os.path.join(folder, "attack_strings_to_probability.json"), "r") as f:
            local_mapping: dict = json.load(f)
        for attack_string in attack_strings:  # local_mapping.items():
            data[attack_string][alpha] = (
                local_mapping[attack_string]["RO"]["scaled"]
                if attack_string in local_mapping
                else 0.0
            )

    filtered: dict[str, dict[float, float]] = {}
    for attack_string in sorted(attack_strings):
        before, after = attack_string.split(".")
        before = before[-before_limit:]
        while before != "" and before[0] == "h":
            before = before[1:]
        if before == "":
            new_attack_string = "."
        else:
            if before[-1] == "a" and after != "" and after[0] == "a":
                after = ""
            new_attack_string = f"{before}.{after}"
        assert new_attack_string in attack_strings, f"{new_attack_string=}"
        if new_attack_string in filtered:
            filtered[new_attack_string] = {
                key: val + data[attack_string][key]
                for key, val in filtered[new_attack_string].items()
            }
        else:
            filtered[new_attack_string] = data[new_attack_string]

    attack_string_to_RO_s = {
        attack_string: sum(entry.values()) for attack_string, entry in filtered.items()
    }
    best_attack_strings = [
        entry[0]
        for entry in sorted(attack_string_to_RO_s.items(), key=lambda x: x[1])[
            -capture:
        ]
    ]

    filtered = {
        attack_string: filtered[attack_string]
        for attack_string in best_attack_strings[::-1]
    }

    path = os.path.join(FIGURES_FOLDER, "chaos.png")
    tkz_path = os.path.join(FIGURES_FOLDER, "chaos.tex")
    plot_data(
        id_to_mapping=filtered,
        # order=["honest", "sm", "fork"],
        id_to_color={
            attack_string: color for attack_string, color in zip(filtered, colors)
        },
        id_to_name={attack_string: attack_string for attack_string in filtered},
        id_to_linestyle={},
        title="CHAOS",
        x_label=STK_X_LABEL,
        left_y_label="Reward (blocks)",
        right_y_label=None,
        to_filename=path,
        tkz_filename=tkz_path,
    )


def grab_first_iteration(
    size_prefix: int, size_postfix: int
) -> dict[str, dict[float, float]]:
    mapping = cache_mapping(
        base_folder=CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    sorted_data = sorted(mapping.items(), key=lambda x: x[0])
    max_alpha = sorted_data[-1][0]
    _, attack_string_mapping = calc_attack_string_mapping(
        alpha=max_alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
    )
    relevant_attack_strings = list(set(attack_string_mapping.values()))
    attack_string_to_graph: dict[str, dict[float, float]] = defaultdict(dict)
    for alpha, folder in sorted_data:
        path = os.path.join(folder, "value_functions", "vf_1.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            entry: dict[str, float] = json.load(f)
        local_data = {eas.split("#")[0]: RO for eas, RO in entry.items()}
        for attack_string in relevant_attack_strings:
            attack_string_to_graph[attack_string][alpha] = local_data[attack_string]
    return attack_string_to_graph


def plot_first_iteration(
    data: dict[str, dict[float, float]], filename: str | None, tkz_filename: str | None
):
    r_as = list(data.keys())[0]
    max_alpha = max(data[r_as])
    latest_sorted = sorted(
        [(attack_string, graph[max_alpha]) for attack_string, graph in data.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    for attack_string, _ in latest_sorted[:8]:
        plt.plot(
            data[attack_string].keys(),
            data[attack_string].values(),
            label=attack_string,
        )
    plt.legend()
    plt.xlabel(STK_X_LABEL)
    plt.ylabel("RO")
    if filename:
        plt.savefig(filename)
    if tkz_filename:
        tikz_save(
            tkz_filename,
            encoding="utf8",
            axis_height="55mm",
            axis_width="8cm",
            strict=False,
            extra_axis_parameters=["font=\small"],
        )
    plt.close()


if __name__ == "__main__":
    data = grab_first_iteration(size_prefix=2, size_postfix=6)
    path = os.path.join(FIGURES_FOLDER, "first_iteration.pdf")
    tkz_path = os.path.join(FIGURES_FOLDER, "first_iteration.tex")
    plot_first_iteration(data, path)
