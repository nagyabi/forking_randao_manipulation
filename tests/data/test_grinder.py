import os
import tempfile
from base.grinder import Grinder
from base.serialize import PickleSerializer


class DummyGrinder(Grinder[list[int]]):
    def __init__(self, out_path: str):
        super().__init__(
            out_path=out_path,
            default_data=[],
            serializer=PickleSerializer(),
            regenerate=False,
            safety_save=True,
            stack=1,
        )

    def _grind(self, **kwargs):
        if kwargs["update"]:
            self._data.append(len(self._data))
            self._data_changed = True


def test_grinder():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "file.pkl")
        dummy_grinder_1 = DummyGrinder(temp_file)
        serializer: PickleSerializer[list[int]] = PickleSerializer()
        assert not os.path.exists(temp_file)
        catched = dummy_grinder_1.start_grinding(update=True)
        assert os.path.exists(temp_file)
        with open(temp_file, "rb") as f:
            data1 = serializer.deserialize(f)
        assert data1 == [0]
        assert data1 == catched

        dummy_grinder_2 = DummyGrinder(temp_file)
        save_name = os.path.join(temp_dir, "file_save_1.pkl")
        assert not os.path.exists(save_name)

        catched2 = dummy_grinder_2.start_grinding(update=True)
        assert os.path.exists(save_name)
        with open(save_name, "rb") as f:
            data_saved = serializer.deserialize(f)
        assert data_saved == [0]
        with open(temp_file, "rb") as f:
            data2 = serializer.deserialize(f)
        assert data2 == [0, 1]
        assert data2 == catched2

        dummy_grinder_3 = DummyGrinder(temp_file)
        save_name2 = os.path.join(temp_dir, "file_save_2.pkl")
        assert not os.path.exists(save_name2)
        assert os.path.exists(save_name)

        catched3 = dummy_grinder_3.start_grinding(update=True)
        assert os.path.exists(save_name2)
        assert not os.path.exists(save_name)
        with open(save_name2, "rb") as f:
            data_saved = serializer.deserialize(f)
        assert data_saved == [0, 1]
        with open(temp_file, "rb") as f:
            data3 = serializer.deserialize(f)
        assert data3 == [0, 1, 2]
        assert data3 == catched3

        dummy_grinder_4 = DummyGrinder(out_path=temp_file)
        save_name3 = os.path.join(temp_dir, "file_save_3.pkl")
        assert not os.path.exists(save_name3)
        assert os.path.exists(save_name2)

        catched4 = dummy_grinder_4.start_grinding(update=False)
        assert not os.path.exists(save_name3)
        assert os.path.exists(save_name2)
        with open(save_name2, "rb") as f:
            data_saved = serializer.deserialize(f)
        assert data_saved == [0, 1]
        with open(temp_file, "rb") as f:
            data4 = serializer.deserialize(f)

        assert data4 == data3
        assert data4 == catched4
