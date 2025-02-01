import json
from os.path import join
import matplotlib.pyplot as plt

from base.helpers import FIGURES_FOLDER

folder_name = join("alpha=0_281-size_prefix=2-size_postfix=6", "value_functions", "vf_10.json")

with open(join("cache", folder_name), "r") as f:
    best_utilities: dict[str, float] = json.load(f)

with open(join("cache3", folder_name), "r") as f:
    approx_utilities: dict[str, float] = json.load(f)

best_utilities = {key: val - best_utilities[".#"] for key, val in best_utilities.items()}
approx_utilities = {key: val - approx_utilities[".#"] for key, val in approx_utilities.items()}

assert list(best_utilities) == list(approx_utilities)

data = [approx_utilities[key] - best_utilities[key] for key in best_utilities if best_utilities[key]]
plt.hist(data, bins=10, edgecolor="black")
plt.savefig(join(FIGURES_FOLDER, "approx_error_65.png"))