from collections import defaultdict
import json
import os

from matplotlib import pyplot as plt
import pandas as pd
from base.helpers import CACHE_FOLDER, FIGURES_FOLDER
from make_figures.base import STK_X_LABEL, cache_mapping, eas_map
import seaborn as sns
try:
    from tikzplotlib import save as tikz_save
except ImportError:
    tikz_save = None

def extract_heatmap_values(
    size_prefix: int, size_postfix: int, iteration: int
) -> dict[float, dict[str, float]]:
    mapping = cache_mapping(
        CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    result: dict[float, dict[str, float]] = defaultdict(dict)
    for alpha, folder in mapping.items():
        if alpha > 0.401:
            continue
        vf_path = os.path.join(folder, "value_functions", f"vf_{iteration}.json")
        if not os.path.exists(vf_path):
            continue
        with open(vf_path, "r") as f:
            entry: dict[str, float] = json.load(f)
        for eas, value in entry.items():
            result[alpha][eas] = value
    return result


def plot_heatmap(data):
    df = pd.DataFrame(data).T
    # with open(os.path.join("Figures", "utility_heatmap.csv"), "w") as f:
    #    df.to_csv(f)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Values"})
    plt.xlabel("Extended attack strings")
    plt.ylabel(STK_X_LABEL)
    plt.title("Utility (adjusted)")
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(FIGURES_FOLDER, "utility_heatmap.png"))
    tkz_filename = os.path.join(FIGURES_FOLDER, "utility_heatmap.tex")
    
    if tikz_save is not None:
        tikz_save(
            tkz_filename,
            encoding="utf8",
            axis_height="55mm",
            axis_width="8cm",
            strict=False,
            extra_axis_parameters=["font=\small"],
        )


if __name__ == "__main__":
    data = extract_heatmap_values(size_prefix=1, size_postfix=2, iteration=10)
    data = {
        alpha: data[alpha]
        for alpha in sorted(list(data))
        if int(round(100 * alpha)) % 2 == 0
    }
    data = {
        alpha: {
            eas_map(eas): value
            for eas, value in entry.items()
            if eas_map(eas) is not None
        }
        for alpha, entry in data.items()
    }
    data = {
        alpha: {
            eas.upper().replace("#", "*"): value - entry["hh.x#hh"]
            for eas, value in sorted(entry.items(), key=lambda x: data[0.2][x[0]])
        }
        for alpha, entry in data.items()
    }
    plot_heatmap(data)
