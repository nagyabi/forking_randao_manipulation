import os
from base.helpers import CACHE_FOLDER, FIGURES_FOLDER
from make_figures.base import (
    STK_X_LABEL,
    cache_mapping,
    extract_infos_attack_strings,
    plot_cumulative_distribution,
)
from theory.method.preeval import calc_attack_string_mapping


def plot_expected_vals_by_as(size_prefix: int, size_postfix: int, iteration: int):
    mapping = cache_mapping(
        CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    data = extract_infos_attack_strings(mapping, iteration)
    base_folder = os.path.join(
        FIGURES_FOLDER, "attack_strings", f"{size_prefix=}-{size_postfix=}-{iteration=}"
    )
    os.makedirs(base_folder, exist_ok=True)
    alphas = list(data["."]["RO"])
    alpha_to_mapping: dict[float, dict[str, str]] = {}
    for alpha in alphas:
        alpha_to_mapping[alpha] = calc_attack_string_mapping(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
        )[1]
    features = list(data["."])
    for attack_string in data:
        graphs: dict[str, dict[float, float]] = {}
        for feature in features:
            graphs[feature] = {}
            for alpha in alphas:
                graphs[feature][alpha] = data[alpha_to_mapping[alpha][attack_string]][
                    feature
                ][alpha]

        keys = ["exp_fork", "exp_regret", "exp_sm"]
        order = [*keys, "honest"]
        datas = {key: graphs[key] for key in keys}
        datas["honest"] = {
            key: graphs["RO"][key] - sum(graph[key] for graph in datas.values())
            for key in datas[keys[0]]
        }

        plot_cumulative_distribution(
            data=datas,
            order=order,
            id_to_color={
                "exp_fork": "green",
                "exp_regret": "red",
                "exp_sm": "yellow",
                "honest": "blue",
            },
            id_to_labels={
                "exp_fork": "Forking",
                "exp_regret": "Regret",
                "exp_sm": "Selfish mixing",
                "honest": "Honest",
            },
            title=attack_string,
            x_label=STK_X_LABEL,
            y_label="Probability (%)",
            to_filename=os.path.join(
                base_folder, attack_string.replace(".", "_") + ".png"
            ),
            tkz_filename=None,  # os.path.join(FIGURES_FOLDER, "exp_vals_additive.tex"),
        )


if __name__ == "__main__":
    plot_expected_vals_by_as(size_prefix=2, size_postfix=6, iteration=25)
