import os
from matplotlib import pyplot as plt
from base.helpers import FIGURES_FOLDER, LATEST_DELIVERY
from base.statistics import CasesEntry, read_delivery_cases
from tikzplotlib import save as tikz_save


def tonum(statuses: str) -> int:
    res = 0
    for s in statuses[::-1]:
        res *= 2
        res += int(s == "p")
    return res


def plot_custom_columns_with_legend_and_grid(
    ax, xs: list[int], ys: list[int], colors: list[str], y_lab: str, title: str
) -> None:
    ax.bar(xs, ys, color=colors)
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel(y_lab)


def create_subplots_with_global_legend_and_titles(
    xs: list[int],
    ys_up: list[int],
    ys_below: list[int],
    colors: list[str],
    title1: str,
    title2: str,
    upper_legend: str,
    below_legend: str,
    y_label_upper: str,
    y_label_below: str,
    global_title: str | None,
    filename: str | None,
    tkz_filename: str | None,
) -> None:
    # Create a figure and two subplots (vertically stacked)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

    # Plot the first dataset in the first subplot
    plot_custom_columns_with_legend_and_grid(
        ax1, xs, ys_up, colors, y_lab=y_label_upper, title=title1
    )

    # Plot the second dataset in the second subplot
    plot_custom_columns_with_legend_and_grid(
        ax2, xs, ys_below, colors, y_lab=y_label_below, title=title2
    )

    # Create a custom legend for the global figure (covering both subplots)

    upper_handler = plt.Line2D([0], [0], color="red", lw=4, label=upper_legend)
    below_handler = plt.Line2D([0], [0], color="blue", lw=4, label=below_legend)

    # Add the global legend at the bottom center of the figure
    fig.legend(
        handles=[below_handler, upper_handler],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.03),
        frameon=False,
        fontsize=10,
    )

    # Add a global title for the entire figure
    if global_title is not None:
        fig.suptitle(global_title, fontsize=16)

    # Adjust the layout to ensure everything fits well
    plt.tight_layout(
        rect=[0, 0.08, 1, 0.95]
    )  # Leave space for the global title and legend

    # Show the plot
    if filename:
        plt.savefig(filename)
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


def plot_case_analyzis(
    entry: CasesEntry, to_filename: str | None, tkz_filename: str | None
):
    data: dict[int, tuple[int, int, str]] = {
        tonum(entry.real_statuses): (
            entry.RO,
            entry.RO + entry.real_statuses.count("m"),
            "red",
        ),
        **{
            tonum(status): (RO, RO + status.count("m"), "blue")
            for status, RO in entry.statuses_to_RO.items()
        },
    }
    points = sorted(data.items(), key=lambda x: x[0])
    xs = [e[0] for e in points]
    ys_RO = [e[1][0] for e in points]
    ys_slots = [e[1][1] for e in points]
    colors = [e[1][2] for e in points]
    create_subplots_with_global_legend_and_titles(
        xs=xs,
        ys_up=ys_RO,
        ys_below=ys_slots,
        colors=colors,
        title1=f"Slots in epoch {entry.epoch + 2} - sacrifice in epoch {entry.epoch}",
        title2=f"Slots in epoch {entry.epoch + 2}",
        upper_legend=f"What {entry.attacker} did",
        below_legend=f"What {entry.attacker} could have done",
        y_label_upper=f"p({entry.epoch + 2}, ., {entry.attacker}) - s({entry.epoch}, ., {entry.attacker})",
        y_label_below=f"p({entry.epoch + 2}, ., {entry.attacker})",
        global_title=None,
        filename=to_filename,
        tkz_filename=tkz_filename,
    )


if __name__ == "__main__":
    delivery = read_delivery_cases(LATEST_DELIVERY)
    matching = [
        entry
        for entry in delivery
        if entry.attack_string.count("h") == 0 and entry.attack_string.count("a") >= 8
    ]
    base_folder = os.path.join(FIGURES_FOLDER, "case_studies")
    os.makedirs(base_folder, exist_ok=True)
    for case in matching:
        plot_case_analyzis(
            to_filename=os.path.join(
                base_folder, f"{case.attacker}_{case.epoch}_study.png"
            ),
            tkz_filename=os.path.join(
                base_folder, f"{case.attacker}_{case.epoch}_study.tex"
            ),
            entry=case,
        )
