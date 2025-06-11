import os

from matplotlib import pyplot as plt
import numpy as np
from base.helpers import CACHE_FOLDER, FIGURES_FOLDER, SLOTS_PER_EPOCH
from make_figures.base import STK_X_LABEL, cache_mapping, extract_infos
try:
    from tikzplotlib import save as tikz_save
except ImportError:
    tikz_save = None

def plot_chain_quality2(size_prefix: int, size_postfix: int, iterations: int):
    mapping = cache_mapping(
        base_folder=CACHE_FOLDER, size_prefix=size_prefix, size_postfix=size_postfix
    )
    infos = extract_infos(alpha_to_folder=mapping, iteration_num=iterations)
    alphas_sorted = sorted(infos["RO"])
    missed = np.array(
        [infos["slots"][alpha] - infos["RO"][alpha] for alpha in alphas_sorted],
        dtype=np.float32,
    )
    forked = np.array(
        [infos["forked_honest_blocks"][alpha] for alpha in alphas_sorted],
        dtype=np.float32,
    )
    data_lower = np.array([0.8] * len(alphas_sorted), dtype=np.float32)
    data_line = 1 - (missed + forked) / SLOTS_PER_EPOCH
    data_between = 1 - missed / SLOTS_PER_EPOCH
    data_upper = np.array([1.0] * len(alphas_sorted), dtype=np.float32)

    plt.figure(figsize=(10, 2))
    plt.fill_between(alphas_sorted, data_lower, data_line, color="blue", alpha=0.6)
    plt.plot(alphas_sorted, data_line, "-", color="yellow")
    plt.fill_between(
        alphas_sorted,
        data_line,
        data_between,
        color="green",
        label="Forked honest blocks",
        alpha=0.6,
    )
    plt.fill_between(
        alphas_sorted,
        data_between,
        data_upper,
        color="red",
        label="Missed adversarial slots",
        alpha=0.6,
    )
    plt.xlabel(STK_X_LABEL)
    plt.ylabel("Chain quality")
    plt.legend(loc="lower left", bbox_to_anchor=(0.05, 0.05), fontsize="xx-small")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "chain_quality_layered.png"))
    tkz_filename = os.path.join(FIGURES_FOLDER, "chain_quality_layered.tex")
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
    plot_chain_quality2(size_prefix=2, size_postfix=6, iterations=25)
