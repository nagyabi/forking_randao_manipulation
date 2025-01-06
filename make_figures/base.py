from collections import defaultdict
from dataclasses import dataclass
import json
import os
from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from scipy.stats import norm
from base.helpers import SLOTS_PER_EPOCH
from tikzplotlib import save as tikz_save


STK_X_LABEL = r"Stakes ($\alpha$)"


@dataclass
class HistogramCfg:
    data: list[float]
    title: str
    bins: int


def plot_4_distributions_vs_normal(
    hist_data: list[HistogramCfg],
    data_label: str,
    title: str,
    to_filename: str | None,
    tkz_filename: str | None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, data in enumerate(hist_data):
        sns.histplot(
            data.data,
            bins=data.bins,
            kde=False,
            stat="density",
            color="skyblue",
            ax=axes[i],
        )

        x = np.linspace(-3, 3, 100)
        axes[i].plot(x, norm.pdf(x, 0, 1), "r-", lw=2)

        axes[i].set_title(data.title)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].grid(True)

    fig.suptitle(title, fontsize=16)

    custom_handles = [
        Line2D([0], [0], color="skyblue", lw=4, label=data_label),  # Histogram
        Line2D([0], [0], color="r", lw=2, label="Std. normal (0,1)"),
    ]  # Normal distribution
    fig.legend(
        handles=custom_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05)
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if to_filename:
        plt.savefig(to_filename, bbox_inches="tight")
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


def cache_mapping(
    base_folder: str, size_prefix: int, size_postfix: int
) -> dict[float, str]:
    """


    Args:
        base_folder (str): caching folder
        size_prefix (int): size of prefix in the epoch string
        size_postfix (int): size of postfix in the epoch string

    Returns:
        dict[float, str]: mapping from alpha stakes to full cache folder names mathing
        size_prefix and size_postfix
    """
    entries = os.listdir(base_folder)
    folders = [
        entry for entry in entries if os.path.isdir(os.path.join(base_folder, entry))
    ]
    folders = [folder for folder in folders if "size_prefix" in folder]

    result: dict[float, str] = {}

    for folder in folders:
        alpha_str, pref_str, postf_str = folder.split("-")
        a, num = alpha_str.split("=")
        if a != "alpha":
            continue
        alpha = float(num.replace("_", "."))
        s, num = pref_str.split("=")
        if s != "size_prefix":
            continue
        if int(num) != size_prefix:
            continue
        s, num = postf_str.split("=")
        if s != "size_postfix":
            continue
        if int(num) != size_postfix:
            continue
        result[alpha] = os.path.join(base_folder, folder)

    return result


def extract_infos(
    alpha_to_folder: dict[float, str], iteration_num: int
) -> dict[str, dict[float, float]]:
    """

    Args:
        alpha_to_folder (dict[str, float]): mapping from alpha to the corresponding
        cache folder (cache_mapping)
        iteration_num (int): number of iterations

    Returns:
        dict[str, dict[float, float]]: mapping from different types of data (RO)
        to mapping from alpha to the value.
    """
    result: dict[str, dict[float, float]] = {}

    for alpha, folder in alpha_to_folder.items():
        expected_values_path = os.path.join(folder, "expected_values.json")
        if not os.path.exists(expected_values_path):
            continue
        with open(expected_values_path, "r") as exp_val_file:
            raw_data: dict[str, dict[str, float]] = json.load(exp_val_file)

        data = {
            int(iter_str): {key: float(val) for key, val in mapping.items()}
            for iter_str, mapping in raw_data.items()
        }

        if iteration_num not in data:
            continue

        for category, value in data[iteration_num].items():
            if category not in result:
                result[category] = {}
            result[category][alpha] = value

    return result


def extract_infos_attack_strings(
    alpha_to_folder: dict[float, str],
    iteration: int,
) -> dict[str, dict[str, dict[float, float]]]:
    """
    Extracts graphs corresponding to attack strings
    """

    result: dict[str, dict[str, dict[float, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for alpha, folder in alpha_to_folder.items():
        attack_strings_path = os.path.join(
            folder, f"attack_strings_to_probability_{iteration}.json"
        )
        if not os.path.exists(attack_strings_path):
            continue
        with open(attack_strings_path, "r") as as_file:
            raw_data: dict[str, dict[str, dict[str, float] | float]] = json.load(
                as_file
            )
        for attack_string, features in raw_data.items():
            for feature, raw in features.items():
                if isinstance(raw, dict):
                    result[attack_string][feature][alpha] = raw["value"]
                else:
                    result[attack_string][feature][alpha] = raw

    return result


def plot_cumulative_distribution(
    data: dict[str, dict[float, float]],
    order: list[str],
    id_to_color: dict[str, str],
    id_to_labels: dict[str, str],
    title: str,
    x_label: str,
    y_label: str,
    to_filename: str | None,
    tkz_filename: str | None,
) -> None:
    # Start plotting
    plt.rc("axes", unicode_minus=False)
    assert len(data) > 0
    assert set(data) == set(order)
    assert all(graph.keys() == data[order[0]].keys() for graph in data.values())
    plt.figure(figsize=(10, 6))

    # Sort alpha values
    alpha_sorted = sorted(data[order[0]])

    # Initialize the cumulative sum of probabilities with zeros
    cum_probs = np.zeros(len(alpha_sorted))

    # Loop through the data keys in the given order and plot cumulative sums
    for key in order:
        probs = np.array([data[key][alpha] for alpha in alpha_sorted])
        cum_probs_next = cum_probs + probs

        plt.fill_between(
            alpha_sorted,
            cum_probs,
            cum_probs_next,
            color=id_to_color[key],
            label=id_to_labels[key],
            alpha=0.6,
        )
        plt.plot(alpha_sorted, cum_probs_next, "-", color=id_to_color[key])

        # Update cumulative sum
        cum_probs = cum_probs_next

    # Final plot settings
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    if tkz_filename:
        tikz_save(
            tkz_filename,
            encoding="utf8",
            axis_height="55mm",
            axis_width="8cm",
            strict=False,
            extra_axis_parameters=["font=\small"],
        )
    if to_filename:
        plt.savefig(to_filename)
    plt.close()


def plot_data(
    id_to_mapping: dict[str, dict[float, float]],
    id_to_name: dict[str, str],
    id_to_color: dict[str, str],
    id_to_linestyle: dict[str, str],
    title: str,
    to_filename: str | None,
    tkz_filename: str | None = None,
    x_label: str = "Stakes (%)",
    left_y_label: str = "Slots",
    right_y_label: str | None = "Effective stakes (%)",
) -> None:
    right_y_scale = 1 / SLOTS_PER_EPOCH
    _, ax1 = plt.subplots(figsize=(8, 8))
    if right_y_label is not None:
        ax2 = ax1.twinx()
    for _id, mappings in id_to_mapping.items():
        name = id_to_name.get(_id, "Unknown")
        color = id_to_color.get(_id, "blue")
        x_values = list(mappings.keys())
        y_values = list(mappings.values())
        ax1.plot(
            x_values,
            y_values,
            label=name,
            color=color,
            linestyle=id_to_linestyle.get(_id, "-"),
        )
        scaled_y_values = [y * right_y_scale for y in y_values]
        if right_y_label is not None:
            ax2.plot(
                x_values,
                scaled_y_values,
                color=color,
                linestyle=id_to_linestyle.get(_id, "-"),
            )
    ax1.set_title(title)
    ax1.set_xlabel(x_label)
    max_perc = max([max(mapping.values()) for mapping in id_to_mapping.values()])
    ax1.set_ylabel(left_y_label)
    ax1.set_ylim(0, max_perc * 1.05)
    if right_y_label is not None:
        ax2.set_ylabel(right_y_label)
        ax2.set_ylim(0, max_perc * right_y_scale * 1.05)
    if x_values:
        ax1.set_xlim(min(x_values), max(x_values))
    ax1.grid(True)
    ax1.set_aspect("auto")
    ax1.legend(loc="upper left", handlelength=4)
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


def eas_map(eas: str) -> Optional[str]:
    """
    Maps from the special case of (m, n) = (1, 2)
    """
    as_mapping = {"ah.a": "ah.a", "aa.": "aa.x", "a.": "ha.x", ".": "hh.x"}
    attack_string, postfix = eas.split("#")
    if attack_string not in as_mapping:
        return None
    return as_mapping[attack_string] + "#" + postfix.rjust(2, "h")
