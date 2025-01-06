from collections import defaultdict
import math
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.patches as mpatches

from base.helpers import FIGURES_FOLDER, SLOTS_PER_EPOCH, Status
from data.file_manager import FileManager

from tikzplotlib import save as tikz_save

YELLOW = (0.96, 0.96, 0.76)
PURPLE = (0.1, 0.413, 0.56)
CYAN = (0.78, 0.2, 0.25)


class UpperTriangleHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        patch = mpatches.Polygon(
            [[x0 + 6, y0 + 6], [x0, y0], [x0, y0 + 6]], facecolor=PURPLE
        )  #  Rectangle([x0, y0], width, height, facecolor='red',
        handlebox.add_artist(patch)
        return patch


class LowerTriangleHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        patch = mpatches.Polygon(
            [[x0 + 6, y0 + 6], [x0, y0], [x0 + 6, y0]], facecolor=CYAN
        )  #  Rectangle([x0, y0], width, height, facecolor='red',
        handlebox.add_artist(patch)
        return patch


class UpperTriangle:
    pass


class LowerTriangle:
    pass


def plot_triangular_grid(
    data_upper: dict[tuple[str, str], int],
    data_lower: dict[tuple[str, str], int],
    cols: list[str],
    rows: list[str],
    title: str,
    x_label: str,
    y_label: str,
    to_filename: str | None,
    tkz_filename: str | None,
    col_mapping: dict[str, str],
) -> None:
    yellow_to_purple = LinearSegmentedColormap.from_list(
        "yellow_to_purple", [YELLOW, PURPLE], N=100
    )
    yellow_to_cyan = LinearSegmentedColormap.from_list(
        "red_to_green", [YELLOW, CYAN], N=100
    )

    upper_max = max(math.log(1 + val) for val in data_upper.values())
    lower_max = max(math.log(1 + val) for val in data_lower.values())

    n_cols = len(cols)
    n_rows = len(rows)

    _, ax = plt.subplots(figsize=(n_cols, n_rows))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            # Draw the square
            rect = plt.Rectangle(
                (j, i),
                1,
                1,
                edgecolor="black",
                color=yellow_to_cyan(
                    (math.log(1 + data_lower[(col, row)])) / lower_max
                ),
            )
            polygon_lower = [(j, i), (j, i + 1), (j + 1, i)]
            ax.add_patch(rect)
            ax.add_patch(
                plt.Polygon(
                    polygon_lower,
                    edgecolor="black",
                    color=yellow_to_purple(
                        (math.log(1 + data_upper[(col, row)])) / upper_max
                    ),
                )
            )

            value_upper = data_upper[(col, row)]
            ax.text(
                j + 0.33,
                i + 0.33,
                str(value_upper) if value_upper else "",
                ha="center",
                va="center",
                fontsize=8,
            )

            value_lower = data_lower[(col, row)]
            ax.text(
                j + 0.67,
                i + 0.67,
                str(value_lower) if value_lower else "",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(
        [col_mapping.get(col_name, col_name) for col_name in cols],
        ha="center",
        fontsize=8,
    )
    ax.xaxis.tick_top()  # Move x-axis labels to the top

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(rows, va="center", fontsize=8)

    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.xaxis.set_label_position("top")

    # Set limits and aspect
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    ax.set_aspect("equal")

    ax.set_title(title, fontsize=18, pad=12)  # Slightly larger title font size

    ax.legend(
        [UpperTriangle(), LowerTriangle()],
        ["All epochs", "Epochs with missed slot(s)"],
        handler_map={
            UpperTriangle: UpperTriangleHandler(),
            LowerTriangle: LowerTriangleHandler(),
        },
        loc="lower right",
        # bbox_to_anchor=(1.05, 1),
        fontsize=8,
        handlelength=1.5,
        handletextpad=0,
    )
    ax.grid(False)

    # Invert y-axis to have the top-left corner as the origin
    plt.gca().invert_yaxis()

    plt.tight_layout()
    if to_filename:
        plt.savefig(to_filename)
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


def extract_missed_heatmap(entities: list[str]) -> tuple[dict, dict]:
    """
    Plots a double heatmap for epochs with by entities.

    Args:
        entities (list[str]): entities to look for
    """

    beaconchain = FileManager.file_manager().beaconchain()
    index_to_entity = FileManager.file_manager().index_to_entity()

    data_all_epochs = defaultdict(int)
    data_missing_epochs = defaultdict(int)

    print("All data read")
    epochs = set(slot // SLOTS_PER_EPOCH for slot in beaconchain)

    epoch = 0
    try:
        for epoch in epochs:
            slot = (epoch + 1) * SLOTS_PER_EPOCH - 1
            candidate = index_to_entity[beaconchain[slot].proposer_index]
            if candidate not in entities:
                continue
            tail_slots = 1
            missed = beaconchain[slot].status == Status.MISSED
            if beaconchain[slot].status == Status.REORGED:
                continue
            slot -= 1

            bad = False
            while index_to_entity[beaconchain[slot].proposer_index] == candidate:
                tail_slots += 1
                missed = missed or beaconchain[slot].status == Status.MISSED
                if beaconchain[slot].status == Status.REORGED:
                    bad = True
                    break
                slot -= 1

            if bad:
                continue

            data_all_epochs[(candidate, str(tail_slots))] += 1
            if missed:
                data_missing_epochs[(candidate, str(tail_slots))] += 1

    except KeyError:
        print(f"Crashed at epoch {epoch}")
    return data_all_epochs, data_missing_epochs


def plot_missed_heatmap(
    entities: list[str],
    entitiy_mapping: dict[str, str],
    data_all_epochs: dict,
    data_missing_epochs: dict,
):
    plot_triangular_grid(
        data_upper=data_all_epochs,
        data_lower=data_missing_epochs,
        cols=entities,
        rows=[str(i) for i in range(1, 9)],
        x_label="Entities",
        y_label="Number of tail slots owned",
        title="Tail slots",
        to_filename=os.path.join(FIGURES_FOLDER, "missed_heatmap.png"),
        tkz_filename=os.path.join(FIGURES_FOLDER, "missed_heatmap.tex"),
        col_mapping=entitiy_mapping,
    )


if __name__ == "__main__":
    path = "dumps/missed_heatmap.pkl"
    if os.path.exists(path):
        with open("dumps/missed_heatmap.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        data = extract_missed_heatmap(
            entities=[
                "Lido",
                "Coinbase",
                "Binance",
                "Kraken",
                "Bitcoin Suisse",
                "DARMA Capital",
                "OKX",
            ],
        )
        with open("dumps/missed_heatmap.pkl", "wb") as f:
            pickle.dump(data, f)

    plot_missed_heatmap(
        entities=[
            "Lido",
            "Coinbase",
            "Binance",
            "Kraken",
            "Bitcoin Suisse",
            "DARMA Capital",
            "OKX",
        ],
        entitiy_mapping={"Bitcoin Suisse": "Bitcoin", "DARMA Capital": "DARMA"},
        data_all_epochs=data[0],
        data_missing_epochs=data[1],
    )
