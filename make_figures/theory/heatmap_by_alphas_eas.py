from collections import defaultdict
import os
import pickle
from base.helpers import CACHE_FOLDER, FIGURES_FOLDER
from make_figures.base import STK_X_LABEL, cache_mapping, eas_map
from theory.method.distribution import RichValuedDistribution
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_heatmap(size_prefix: int, size_postfix: int, iteration: int):
    mapping = cache_mapping(
        CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    result: dict[float, dict[str, float]] = defaultdict(dict)
    for alpha, folder in mapping.items():
        rich_dist_path = os.path.join(
            folder, "rich_distribution", f"rd_{iteration}.pkl"
        )
        if not os.path.exists(rich_dist_path):
            continue
        with open(rich_dist_path, "rb") as f:
            distributions: dict[str, RichValuedDistribution]
            _, distributions = pickle.load(f)
        for eas, dist in distributions.items():
            result[alpha][eas] = dist.distribution[0].expected_value_of_vars[0]

    return result


def plot_heatmap(data):
    df = pd.DataFrame(data).T
    # with open(os.path.join(FIGURES_FOLDER, "heatmap_eas_alphas_immediate_reward.csv"), "w") as f:
    #    df.to_csv(f)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Values"})
    plt.xlabel("Extended attack strings")
    plt.ylabel(STK_X_LABEL)
    plt.title("Expected immediate rewards")
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(FIGURES_FOLDER, "test_heatmap.png"))


if __name__ == "__main__":
    data = extract_heatmap(size_prefix=1, size_postfix=2, iteration=10)
    data = {alpha: val for alpha, val in data.items() if alpha < 0.41}
    keys = sorted(list(data))
    data_sorted = {key: data[key] for key in keys}

    data_sorted = {
        alpha: {
            key: round(val, 2)
            for key, val in sorted(entry.items(), key=lambda x: x[0].split("#")[0])
        }
        for alpha, entry in data_sorted.items()
    }

    data_sorted = {
        alpha: {
            eas_map(eas): val for eas, val in entry.items() if eas_map(eas) is not None
        }
        for alpha, entry in data_sorted.items()
        if int(round(alpha * 100)) % 2 == 0
    }

    plot_heatmap(data=data_sorted)
