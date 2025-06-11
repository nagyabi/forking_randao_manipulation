from dataclasses import dataclass
import os

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from base.helpers import FIGURES_FOLDER, LATEST_DELIVERY, SLOTS_PER_EPOCH
from data.hypothesis.proposed import (
    MIN_CASES,
    normalize_proposed,
    observed_vs_expected_interpol,
)
from make_figures.base import HistogramCfg, plot_4_distributions_vs_normal

try:
    from tikzplotlib import save as tikz_save
except ImportError:
    tikz_save = None

@dataclass
class BarCfg:
    indexed: list[tuple[tuple[int, int], int]]
    xs: np.ndarray
    ys: np.ndarray
    title: str


def plot_4_distributions_vs_expected(
    bar_data: list[BarCfg],
    data_label: str,
    title: str,
    to_filename: str | None,
    tkz_filename: str | None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, data in enumerate(bar_data):
        # sns.histplot(data.data, bins=data.bins, kde=False, stat="density", color='skyblue', ax=axes[i])
        for base, height in data.indexed:
            left = base[0]  # Starting point of the bar
            width = base[1] - base[0]  # Width of the bar
            axes[i].bar(
                left,
                height / width,
                width=width,
                align="edge",
                color="skyblue",
                edgecolor="black",
            )
        axes[i].plot(data.xs, data.ys, "r-", lw=2, label="Expected")

        axes[i].set_xlim(0, max(data.xs))
        axes[i].set_title(data.title)
        axes[i].set_xlabel("Slots")
        axes[i].set_ylabel("Number of epochs")
        axes[i].grid(True)

    fig.suptitle(title, fontsize=16)

    custom_handles = [
        Line2D([0], [0], color="skyblue", lw=4, label=data_label),  # Histogram
        Line2D([0], [0], color="r", lw=2, label="Expected"),
    ]  # Normal distribution
    fig.legend(
        handles=custom_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05)
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if to_filename:
        plt.savefig(to_filename, bbox_inches="tight")
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


def plot_4_proposed_slots(delivery_path: str, entities_to_bins: dict[str, int]) -> None:
    normalized = normalize_proposed(
        delivery_path=delivery_path, entities=list(entities_to_bins)
    )
    to_filename = os.path.join(FIGURES_FOLDER, "proposed_by_entities_mult.png")
    tkz_filename = os.path.join(FIGURES_FOLDER, "proposed_by_entities_mult.tex")
    hist_data = [
        HistogramCfg(
            data=data,
            title=entity,
            bins=entities_to_bins[entity],
        )
        for entity, data in normalized.items()
    ]

    plot_4_distributions_vs_normal(
        bar_data=hist_data,
        data_label="Normalized slots",
        title="Reward when entities proposed",
        to_filename=to_filename,
        tkz_filename=tkz_filename,
    )


def group(act: np.ndarray):
    while act[-1] == 0:
        act = act[:-1]

    def helper(
        indexed: list[tuple[tuple[int, int], int]]
    ) -> list[tuple[tuple[int, int], int]]:
        h_cum = 0
        hist_res = []
        need_to = True
        curr_fr, curr_to = SLOTS_PER_EPOCH + 1, -1
        for (fr, to), h in indexed:
            h_cum += h
            curr_fr = min(curr_fr, fr)
            curr_to = max(curr_to, to)
            if h_cum >= MIN_CASES:
                hist_res.append(((curr_fr, curr_to), h_cum))
                h_cum = 0
                need_to = False
                curr_fr, curr_to = SLOTS_PER_EPOCH + 1, -1
            else:
                need_to = True
        if need_to:
            hist_res.append(((curr_fr, curr_to), h_cum))
        return hist_res

    indexed = [((i, i + 1), e) for i, e in enumerate(act)]
    indexed = helper(indexed[::-1])
    indexed = helper(indexed[::-1])
    return indexed


def plot_4_ent_binom(delivery_path: str, entities: list[str]):
    data = observed_vs_expected_interpol(
        delivery_path=delivery_path, entities=entities, proposed=True, density=0.05
    )
    bar_data: list[BarCfg] = []
    for entity, (act, (xs, ys)) in data.items():
        indexed = group(act)

        max_x = indexed[-1][0][1]

        while xs[-1] > max_x:
            xs = xs[:-1]
            ys = ys[:-1]
        bar_data.append(
            BarCfg(
                indexed=indexed,
                xs=0.5 + xs,
                ys=ys * np.sum(act),
                title=entity,
            )
        )
    to_filename = os.path.join(FIGURES_FOLDER, "proposed_by_entities_mult_binom.png")
    tkz_filename = os.path.join(FIGURES_FOLDER, "proposed_by_entities_mult_binom.tex")

    plot_4_distributions_vs_expected(
        bar_data=bar_data,
        data_label="Number of epochs",
        title="Reward when entities proposed",
        to_filename=to_filename,
        tkz_filename=tkz_filename,
    )


if __name__ == "__main__":
    plot_4_ent_binom(
        delivery_path=LATEST_DELIVERY,
        entities=["Lido", "Coinbase", "Binance", "Kraken"],
    )
    exit()
    plot_4_proposed_slots(
        delivery_path=LATEST_DELIVERY,
        entities_to_bins={"Lido": 25, "Coinbase": 12, "Binance": 10, "Kraken": 10},
    )
    # proposed_slots(delivery_path=LATEST_DELIVERY, entities=["Kraken"])
