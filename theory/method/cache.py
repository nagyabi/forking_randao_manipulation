import json
import os
import pickle
import re
from typing import Any, Optional
import numpy as np

from base.helpers import CACHE_FOLDER
from data.serialize import QuantizedEASSerializer
from theory.method.quant.base import QuantizedEAS
from theory.method.distribution import RichValuedDistribution
from theory.method.quant.quantize import Quantizer


def cache_path(alpha: np.float64, size_prefix: int, size_postfix: int):
    alpha_string = str(np.round(alpha, decimals=5)).replace(".", "_")
    return f"alpha={alpha_string}-{size_prefix=}-{size_postfix=}"


def parse_path(folder_name: str) -> tuple[np.float64, int, int]:
    alpha, size_prefix, size_postfix = folder_name.split("-")
    alpha = np.float64(alpha)
    size_prefix = int(size_prefix)
    size_postfix = int(size_postfix)
    return alpha, size_prefix, size_postfix


class Cacher:
    def __init__(
        self,
        alpha: np.float64,
        size_prefix: int,
        size_postfix: int,
        default: dict[str, np.float64],
        base_folder: str = CACHE_FOLDER,
        should_exists: bool = False,
    ):
        folder_name = cache_path(
            alpha=alpha, size_prefix=size_prefix, size_postfix=size_postfix
        )
        if base_folder is None:
            base_folder = CACHE_FOLDER
        self.folder_name = os.path.join(base_folder, folder_name)
        if should_exists:
            assert os.path.exists(
                self.folder_name
            ), f"No cache corresponds to {alpha=} {size_prefix=} {size_postfix=}"
        self.default = default
        self.quant_serializer = QuantizedEASSerializer()

        self.value_function_folder = os.path.join(self.folder_name, "value_functions")
        os.makedirs(self.value_function_folder, exist_ok=True)
        self.rich_distribution_folder = os.path.join(
            self.folder_name, "rich_distribution"
        )
        os.makedirs(self.rich_distribution_folder, exist_ok=True)
        self.quant_distribution_folder = os.path.join(self.folder_name, "quant")
        os.makedirs(self.quant_distribution_folder, exist_ok=True)

    def __path_to_versions(self, folder: str, pattern: str) -> dict[str, int]:
        dirs = [os.path.join(folder, entry) for entry in os.listdir(folder)]
        files = [file for file in dirs if os.path.isfile(file)]
        good_files = [
            file for file in files if re.match(pattern, os.path.split(file)[1])
        ]

        path_to_version: dict[str, int] = {
            file: int(re.search(pattern, os.path.split(file)[1]).group(1))
            for file in good_files
        }
        return path_to_version

    def __latest(
        self, folder: str, pattern: str, iteration: int
    ) -> tuple[Optional[str], int]:
        """

        Returns:
            tuple[Optional[str], int]: The first value from the tuple is the previously file with the highest version/iteration.
            The second member is the number of the current version
        """
        path_to_version = self.__path_to_versions(folder=folder, pattern=pattern)
        if len(path_to_version) == 0 or iteration == 0:
            return None, 0
        matching = [
            path for path, version in path_to_version.items() if version == iteration
        ]
        if len(matching) > 0:
            return matching[0], iteration + 1
        biggest_iteration = max(path_to_version.items(), key=lambda x: x[1])
        return biggest_iteration[0], biggest_iteration[1] + 1

    def value_function(self, iteration: int) -> tuple[int, dict[str, np.float64]]:
        path, iter = self.__latest(
            folder=self.value_function_folder,
            pattern=r"^vf_(\d+)\.json$",
            iteration=iteration,
        )
        if path is None:
            assert iter == 0
            return 1, dict(self.default)
        with open(path, "r") as f:
            data = json.load(f)

        return iter, {key: np.float64(val) for key, val in data.items()}

    def value_function_by_iter(self, iteration: int) -> Optional[dict[str, np.float64]]:
        path = os.path.join(self.value_function_folder, f"vf_{iteration}.json")
        if not os.path.exists(path=path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return {key: np.float64(val) for key, val in data.items()}

    def push_value_function(
        self, iter: int, value_function: dict[str, np.float64]
    ) -> None:
        with open(
            os.path.join(self.value_function_folder, f"vf_{iter}.json"), "w"
        ) as f:
            json.dump(value_function, f)

    def rich_distributions(
        self, iteration: int
    ) -> Optional[tuple[dict[str, list[str]], dict[str, RichValuedDistribution]]]:
        path_to_version = self.__path_to_versions(
            folder=self.rich_distribution_folder, pattern=r"^rd_(\d+)\.pkl$"
        )
        version_to_path = {version: path for path, version in path_to_version.items()}
        if iteration not in version_to_path:
            return None
        path = version_to_path[iteration]
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def push_rich_distributions(
        self,
        meanings: dict[str, list[str]],
        r_dists: dict[str, RichValuedDistribution],
        iteration: int,
    ) -> None:
        fname = f"rd_{iteration}.pkl"
        path = os.path.join(self.rich_distribution_folder, fname)
        with open(path, "wb") as f:
            pickle.dump((meanings, r_dists), f)

    def expected_values(self) -> dict[int, dict[str, np.float64]]:
        path = os.path.join(self.folder_name, "expected_values.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        try:
            return {
                int(key): {k: np.float64(v) for k, v in val.items()}
                for key, val in data.items()
            }
        except Exception:
            return {}

    def push_expected_values(
        self, expected_values: dict[int, dict[str, np.float64]]
    ) -> None:
        path = os.path.join(self.folder_name, "expected_values.json")
        with open(path, "w") as f:
            json.dump(expected_values, f)

    def push_attack_strings_to_probabilities(
        self, attack_strings_to_probability: dict[str, dict[str, Any]], iterations: int
    ) -> None:
        path = os.path.join(
            self.folder_name, f"attack_strings_to_probability_{iterations}.json"
        )
        with open(path, "w") as f:
            json.dump(attack_strings_to_probability, f)

    def already_quant(self, iteration: int) -> bool:
        quant_mapping_path = os.path.join(
            self.quant_distribution_folder, f"quant_eass_{iteration}.json"
        )
        return os.path.exists(quant_mapping_path)

    def get_quant(
        self, iteration: int
    ) -> tuple[
        dict[str, str], dict[str, QuantizedEAS], dict[str, tuple[int, dict[str, int]]]
    ]:
        mapping_path = os.path.join(self.quant_distribution_folder, "eas_mapping.json")
        with open(mapping_path, "r") as mapping_file:
            eas_mapping = json.load(mapping_file)
        quant_mapping_path = os.path.join(
            self.quant_distribution_folder, f"quant_eass_{iteration}.json"
        )
        with open(quant_mapping_path, "r") as quant_file:
            eas_to_quant = self.quant_serializer.deserialize(file=quant_file)
        quant_postfix_mapping_path = os.path.join(
            self.quant_distribution_folder, f"postfix_mapping_{iteration}.json"
        )
        with open(quant_postfix_mapping_path, "r") as postf_mapping_file:
            mapping_by_eas_postf = json.load(postf_mapping_file)
        return eas_mapping, eas_to_quant, mapping_by_eas_postf

    def push_quant(self, quantizer: Quantizer, iteration: int) -> None:
        mapping_path = os.path.join(self.quant_distribution_folder, "eas_mapping.json")
        with open(mapping_path, "w") as mapping_file:
            json.dump(quantizer.eas_mapping, mapping_file)
        quant_mapping_path = os.path.join(
            self.quant_distribution_folder, f"quant_eass_{iteration}.json"
        )
        with open(quant_mapping_path, "w") as quant_file:
            self.quant_serializer.serialize(
                data=quantizer.eas_to_quant, file=quant_file
            )
        quant_postfix_mapping_path = os.path.join(
            self.quant_distribution_folder, f"postfix_mapping_{iteration}.json"
        )
        with open(quant_postfix_mapping_path, "w") as postf_mapping_file:
            json.dump(quantizer.mapping_by_eas_postf, postf_mapping_file)
