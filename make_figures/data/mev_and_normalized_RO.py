from collections import defaultdict
import math
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
from base.helpers import FIGURES_FOLDER, SLOTS_PER_EPOCH
from data.process.serialize import ReorgEntrySerializer
from scipy.stats import linregress
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

try:
    from tikzplotlib import save as tikz_save
except:
    tikz_save = None


def collect_after_reward_data() -> list[tuple[str, float, float]]:
    reorg_serializer = ReorgEntrySerializer()
    with open("data/processed_data/reorgs.json", "r") as f:
        reorg_entries = reorg_serializer.deserialize(f)
    reorg_entries = list(reorg_entries.values())
    with open("data/jsons/mev.pkl", "rb") as f:
        mevs = pickle.load(f)
    data = [
        (
            reorg_entry.entity,
            (reorg_entry.slots - SLOTS_PER_EPOCH * reorg_entry.stake)
            / math.sqrt(SLOTS_PER_EPOCH * reorg_entry.stake * (1 - reorg_entry.stake)),
            mevs[reorg_entry.block_number],
        )
        for reorg_entry in reorg_entries
        if mevs[reorg_entry.block_number] > 10**7
    ]
    return data


def plot_MEV_RO(
    data: list[tuple[str, float, float]],
    id_to_color: dict[str, str],
    title: str,
    x_label: str,
    y_label: str,
    to_filename: str | None,
    tkz_filename: str | None,
):
    colors = np.array([id_to_color[entry[0]] for entry in data])
    x = np.array([entry[1] for entry in data])
    y = np.array([math.log10(entry[2]) for entry in data])

    plt.figure()
    plt.scatter(x, y, c=colors)  # Use scatter plot for dots

    for color in np.unique(colors):
        mask = colors == color
        x_group = x[mask]
        log_y_group = y[mask]
        slope, intercept, _, _, _ = linregress(x_group, log_y_group)
        plt.plot(
            x_group,
            slope * x_group + intercept,
            color=color,
            label=f"Trend line ({color})",
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(
        handles=[
            mpatches.Patch(color=color, label=id) for id, color in id_to_color.items()
        ]
    )

    # Show the plot
    if to_filename:
        plt.savefig(to_filename)
    if tkz_filename and tikz_save is not None:
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
    data = collect_after_reward_data()

    id_to_color = {"Lido": "red", "Binance": "blue", "Coinbase": "green"}
    data = [entry for entry in data if entry[0] in id_to_color]
    plot_MEV_RO(
        data=data,
        id_to_color=id_to_color,
        title=f"Slots and MEV",
        x_label="Normalized slots",
        y_label="MEV (log)",
        to_filename=os.path.join(FIGURES_FOLDER, "MEV_RO.png"),
        tkz_filename=os.path.join(FIGURES_FOLDER, "MEV_RO.tex"),
    )
    entity_to_data: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for entity, x, y in data:
        if y > 10**7:
            entity_to_data[entity].append((x, y))
    print({key: len(val) for key, val in entity_to_data.items()})
    for entity, entry in entity_to_data.items():
        xs, ys_log = np.array([bl[0] for bl in entry]), np.log(
            np.array([bl[1] for bl in entry], dtype=np.float32)
        )
        pearson_corr, _ = pearsonr(xs, ys_log)
        print(f"{entity} => {pearson_corr}")
