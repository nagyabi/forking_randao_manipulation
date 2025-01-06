import json
import os

from base.helpers import CACHE_FOLDER, FIGURES_FOLDER, SLOTS_PER_EPOCH
from make_figures.base import STK_X_LABEL, cache_mapping, extract_infos, plot_data


def plot_theory():
    mapping = cache_mapping(base_folder=CACHE_FOLDER, size_prefix=2, size_postfix=6)
    infos = extract_infos(alpha_to_folder=mapping, iteration_num=25)
    keys = list(infos)
    for key in keys:
        if "prob" in key or "exp" in key:
            del infos[key]

    infos["id"] = {key: key * SLOTS_PER_EPOCH for key in infos["RO"]}

    with open(os.path.join(CACHE_FOLDER, "selfish_mixing_32.json"), "r") as f:
        selfish_mixing = {float(key): float(val) for key, val in json.load(f).items()}
    infos["selfish_mixing"] = selfish_mixing

    id_to_name = {
        "RO": "Adversarial proposed blocks",
        "slots": "Adversarial slots",
        "forked_honest_blocks": "Forked honest blocks",
        "selfish_mixing": "AW24 (selfish mixing)",
        "id": "Honest",
    }
    id_to_color = {
        "RO": "red",
        "slots": "purple",
        "forked_honest_blocks": "green",
        "selfish_mixing": "brown",
        "id": "blue",
    }

    infos = {
        plotname: {key: mapping[key] for key in sorted(mapping)}
        for plotname, mapping in infos.items()
    }
    id_to_linestyle = {
        "RO": "-",
        "slots": ":",
        "forked_honest_blocks": "--",
        "selfish_mixing": "-.",
        "id": (0, (10, 5)),
    }

    plot_data(
        id_to_mapping=infos,
        id_to_name=id_to_name,
        id_to_color=id_to_color,
        id_to_linestyle=id_to_linestyle,
        title="Strategic manipulations",
        to_filename=os.path.join(FIGURES_FOLDER, "expected_values.png"),
        tkz_filename=os.path.join(FIGURES_FOLDER, "expected_values.tex"),
        x_label=STK_X_LABEL,
        right_y_label=r"Effective stakes ($\alpha$)",
    )


if __name__ == "__main__":
    plot_theory()
