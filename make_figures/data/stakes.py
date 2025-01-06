import os
from base.helpers import BIG_ENTITIES, FIGURES_FOLDER, LATEST_DELIVERY
from base.statistics import read_delivery_cases
from make_figures.base import STK_X_LABEL, plot_data


def plot_stakes(path: str, entities: list[str]):
    data = read_delivery_cases(path=path)
    data = sorted(data, key=lambda x: x.epoch)
    base_folder = os.path.join(FIGURES_FOLDER, "stakes")
    os.makedirs(base_folder, exist_ok=True)

    for entity in entities:
        stakes = {
            entry.epoch: entry.stake for entry in data if entry.attacker == entity
        }
        plot_data(
            id_to_mapping={"stakes": stakes},
            id_to_name={"stakes": f"{entity}"},
            id_to_color={"stakes": "red"},
            title=f"{entity} stakes",
            to_filename=os.path.join(base_folder, f"{entity}.png"),
            tkz_filename=os.path.join(base_folder, f"{entity}.tex"),
            x_label="Epoch",
            left_y_label=STK_X_LABEL,
            right_y_label=None,
        )


if __name__ == "__main__":
    plot_stakes(LATEST_DELIVERY, entities=BIG_ENTITIES)
