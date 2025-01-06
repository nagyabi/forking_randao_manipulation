import os
import numpy as np
from base.helpers import FIGURES_FOLDER
from make_figures.base import STK_X_LABEL, plot_data
from theory.method.one_shot import OneShotCalculator


def one_shot(
    size_prefix: int,
    size_postfix: int,
) -> None:
    calc = OneShotCalculator(size_prefix=size_prefix, size_postfix=size_postfix)
    alpha = 0.001
    data = {}
    while alpha < 0.5:
        data[alpha] = calc.calc_prob(alpha=np.float64(alpha))
        alpha += 0.005

    data = {float(key): val for key, val in data.items()}
    id = {key: key for key in data}
    plot_data(
        id_to_mapping={
            "man": data,
            "id": id,
        },
        id_to_name={
            "man": "Manipulated",
            "id": "Honest",
        },
        id_to_color={
            "man": "red",
            "id": "blue",
        },
        id_to_linestyle={
            "man": "-",
            "id": (0, (10, 5)),
        },
        title="Probability of obtaining a target slot",
        to_filename=os.path.join(
            FIGURES_FOLDER, f"one_shot_{size_prefix}_{size_postfix}.png"
        ),
        tkz_filename=os.path.join(
            FIGURES_FOLDER, f"one_shot_{size_prefix}_{size_postfix}.tex"
        ),
        x_label=STK_X_LABEL,
        left_y_label="Probability of getting the slot",
        right_y_label=None,
    )


if __name__ == "__main__":
    one_shot(size_postfix=8, size_prefix=2)
