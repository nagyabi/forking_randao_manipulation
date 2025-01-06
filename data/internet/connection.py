from abc import ABC, abstractmethod
from base.beaconchain import BlockData


class BeaconChainConnection(ABC):
    @abstractmethod
    def slot_data(self, slot: int) -> BlockData:
        pass

    @abstractmethod
    def epoch_data(self, epoch: int) -> dict[int, BlockData]:
        pass

    @abstractmethod
    def latest_epoch(self) -> int:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_waiting_time(self) -> float:
        pass
