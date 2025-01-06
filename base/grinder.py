from abc import ABC, abstractmethod
from copy import deepcopy
import os
import re
import shutil
from typing import Generic, Optional

from base.serialize import DataType, Serializer


class Grinder(ABC, Generic[DataType]):
    def __init__(
        self,
        out_path: str,
        default_data: DataType,
        serializer: Serializer[DataType],
        regenerate: bool = False,
        safety_save: bool = True,
        stack: int = 1,
    ):
        self._data: DataType = deepcopy(default_data)
        self._data_changed: bool = False
        self.__out_path = out_path
        self.__serializer = serializer
        self.__stack = stack
        self.__safety_save = safety_save
        out_folder, _ = os.path.split(out_path)
        os.makedirs(out_folder, exist_ok=True)
        if os.path.exists(out_path) and not regenerate:
            print("Found already existing out file")
            format = "b" if self.__serializer.is_binary() else ""
            with open(self.__out_path, f"r{format}") as f:
                self._data = self.__serializer.deserialize(file=f)

    def __update_file(self):
        if self._data_changed:
            if os.path.exists(self.__out_path) and self.__safety_save:
                self.__safety_saving()
            print("Start updating...")
            format = "b" if self.__serializer.is_binary() else ""
            with open(self.__out_path, f"w{format}") as f:
                self.__serializer.serialize(self._data, f)
            print("Done updating!")
        else:
            print("Skipped updating (no change registered)")

    @abstractmethod
    def _grind(self, **kwargs):
        """
        Data specific function that might start a lot of api calls or any other long time processes.
        """

    def _before_end(self):
        """
        This function will run before the final updating the file
        """

    def __safety_saving(self):
        assert self.__stack > 0
        directory, filename = os.path.split(self.__out_path)
        files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]

        name, extension = filename.split(".")
        pattern = rf"{name}_save_(\d+)\.{extension}"
        saves: dict[int, str] = {}
        for file in files:
            match = re.match(pattern, file)
            if match:
                num = int(match.group(1))
                saves[num] = file
        version = 1 if len(saves) == 0 else 1 + max(saves)
        safety_filename = f"{name}_save_{version}.{extension}"
        saves[version] = safety_filename
        versions = sorted(list(saves))
        old_versions = versions[: -self.__stack]
        old_files = [saves[version] for version in old_versions]
        assert safety_filename not in old_files
        print(f"Making safety save named {safety_filename}")
        shutil.copy(self.__out_path, os.path.join(directory, safety_filename))
        for old_file in old_files:
            print(f"Deleting {os.path.join(directory, old_file)}")
            os.remove(os.path.join(directory, old_file))

    def start_grinding(self, **kwargs) -> Optional[DataType]:
        normal = True
        try:
            self._grind(**kwargs)
        except KeyboardInterrupt:
            print("User stopped the program")
            normal = False
        except Exception as e:
            print("Exception occured")
            self.__update_file()
            raise e  # Reraising the exception after saving the progress
        self._before_end()
        self.__update_file()
        if normal:
            return self._data
