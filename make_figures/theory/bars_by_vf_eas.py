from collections import defaultdict
import json
import os

from matplotlib import pyplot as plt
from base.helpers import CACHE_FOLDER, FIGURES_FOLDER
from make_figures.base import cache_mapping, eas_map
import matplotlib.cm as cm


def extract_vf(size_prefix: int, size_postfix: int, alpha: float):
    mapping = cache_mapping(
        CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    closest = min(mapping, key=lambda x: abs(x - alpha))
    assert abs(closest - alpha) < 1e-4, f"No data found around {alpha}"
    folder = mapping[closest]
    result: dict[str, dict[int, float]] = defaultdict(dict)
    vf_folder = os.path.join(folder, "value_functions")
    fnames = os.listdir(vf_folder)
    paths = [os.path.join(vf_folder, f) for f in fnames if f.startswith("vf_")]
    for path in paths:
        num = int(path.split("vf_")[1].split(".")[0])
        with open(path, "r") as f:
            entry: dict[str, float] = json.load(f)
        for eas, value in entry.items():
            result[eas][num] = value

    return result


def plot_heatmap_bars(data: dict[str, float]):
    labels = list(data.keys())
    values = list(data.values())

    norm = plt.Normalize(min(values), max(values))
    colors = cm.viridis(norm(values))

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color=colors)

    plt.xlabel("Extended attack strings")
    plt.ylabel("Utility")
    plt.title("Utility function (adjusted)")

    plt.xticks(rotation=45, ha="right")

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="Utility")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "bars_eas_iter.png"))


if __name__ == "__main__":
    data = extract_vf(size_prefix=1, size_postfix=2, alpha=0.2)

    data = {
        eas_map(eas): {key: entry[key] - data[".#"][key] for key in sorted(list(entry))}
        for eas, entry in data.items()
        if eas_map(eas) is not None
    }

    custom = {key: val[10] for key, val in data.items()}
    custom = {
        key: val
        for key, val in sorted(custom.items(), key=lambda x: x[1])
        if not key.startswith("ah.h")
    }
    plot_heatmap_bars(custom)
