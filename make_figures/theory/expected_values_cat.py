import os
from base.helpers import CACHE_FOLDER, FIGURES_FOLDER
from make_figures.base import (
    STK_X_LABEL,
    cache_mapping,
    extract_infos,
    plot_cumulative_distribution,
)


def plot_expected_values_by_categories():
    mapping = cache_mapping(base_folder=CACHE_FOLDER, size_prefix=2, size_postfix=6)
    infos = extract_infos(alpha_to_folder=mapping, iteration_num=25)

    keys = ["exp_fork", "exp_regret", "exp_sm"]
    order = [*keys, "honest"]
    datas = {key: infos[key] for key in keys}
    datas["honest"] = {
        key: infos["RO"][key] - sum(graph[key] for graph in datas.values())
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
        title="Actions adding up to total reward",
        x_label=STK_X_LABEL,
        y_label="Probability (%)",
        to_filename=os.path.join(FIGURES_FOLDER, "exp_vals_additive.png"),
        tkz_filename=os.path.join(FIGURES_FOLDER, "exp_vals_additive.tex"),
    )


if __name__ == "__main__":
    plot_expected_values_by_categories()
