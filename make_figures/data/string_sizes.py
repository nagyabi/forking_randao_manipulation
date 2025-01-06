import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from base.helpers import FIGURES_FOLDER
from make_figures.base import STK_X_LABEL
from theory.method.preeval import (
    calc_attack_string_mapping,
    calc_eas_mapping,
    calc_postfix_mapping,
)


def get_mappings(alphas: list[float], size_prefix: int, size_postfix: int):
    attack_string_sizes: dict[float, int] = {}
    eas_string_sizes: dict[float, int] = {}
    for alpha in tqdm(alphas):
        _, attack_string_mapping = calc_attack_string_mapping(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
        )
        eas_mapping = calc_eas_mapping(
            attack_string_mapping=attack_string_mapping,
            postfix_mapping=calc_postfix_mapping(attack_string_mapping),
        )
        attack_string_sizes[alpha] = len(set(attack_string_mapping.values()))
        eas_string_sizes[alpha] = len(set(eas_mapping.values()))
    return attack_string_sizes, eas_string_sizes


def plot_string_sizes(size_prefix: int, postfix_to_color: dict[int, str]):
    alphas = []
    a = 0.001
    while a < 0.5:
        alphas.append(a)
        a += 0.01
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for size_postfix, color in postfix_to_color.items():
        as_sizes, eas_sizes = get_mappings(
            alphas=alphas, size_prefix=size_prefix, size_postfix=size_postfix
        )
        ax1.plot(
            as_sizes.keys(),
            as_sizes.values(),
            color=color,
            label=f"postfix={size_postfix}",
        )
        ax2.plot(
            eas_sizes.keys(),
            eas_sizes.values(),
            color=color,
            label=f"postfix={size_postfix}",
        )

    ax1.set_title("Attack strings")
    ax2.set_title("Extended attack strings")
    ax1.set_xlabel(STK_X_LABEL)
    ax2.set_xlabel(STK_X_LABEL)
    ax1.set_ylabel("Relevant strings")
    ax2.set_ylabel("Relevant strings")
    ax1.legend()
    ax2.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(FIGURES_FOLDER, "string_sizes.png"))
    plt.close()


if __name__ == "__main__":
    plot_string_sizes(size_prefix=2, postfix_to_color={6: "blue", 7: "red", 8: "green"})
