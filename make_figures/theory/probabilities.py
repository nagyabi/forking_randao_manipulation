import os
from base.helpers import CACHE_FOLDER, FIGURES_FOLDER
from make_figures.base import (
    STK_X_LABEL,
    cache_mapping,
    extract_infos,
    plot_cumulative_distribution,
)


def plot_probabilities(size_prefix=2, size_postfix=6, iteration_num=25):
    mapping = cache_mapping(
        base_folder=CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    datas = extract_infos(alpha_to_folder=mapping, iteration_num=iteration_num)

    order = ["fork_prob", "regret_prob", "sm_prob"]
    datas = {key: datas[key] for key in order}
    datas["honest"] = {
        key: 1 - sum(graph[key] for graph in datas.values()) for key in datas[order[0]]
    }
    order.append("honest")

    datas = {
        id: {key: 100 * val for key, val in graph.items()}
        for id, graph in datas.items()
    }

    plot_cumulative_distribution(
        data=datas,
        order=order,
        id_to_color={
            "fork_prob": "green",
            "regret_prob": "red",
            "sm_prob": "yellow",
            "honest": "blue",
        },
        id_to_labels={
            "fork_prob": "Forking",
            "regret_prob": "Regret",
            "sm_prob": "Selfish mixing",
            "honest": "Honest",
        },
        title="Distribution of actions in the optimal policy",
        x_label=STK_X_LABEL,
        y_label="Probability (%)",
        to_filename=os.path.join(FIGURES_FOLDER, "probabilities.png"),
        tkz_filename=os.path.join(FIGURES_FOLDER, "probabilities.tex"),
    )


if __name__ == "__main__":
    plot_probabilities()
