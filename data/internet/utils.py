from abc import ABC, abstractmethod
import time


def safe_function(function, **kwargs):
    for _ in range(5):
        try:
            return function(**kwargs)
        except Exception as e:
            exc = e
            print("Safe function failed, retrying...")
        time.sleep(6)
    print("Safe function failed")
    raise exc


class APITester(ABC):
    @abstractmethod
    def test_api_keys(self, api_keys: list[str]) -> dict[str, bool]:
        pass
