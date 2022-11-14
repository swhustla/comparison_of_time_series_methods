from typing import Protocol, TypeVar

Data = TypeVar("Data")

class Impute(Protocol[Data]):
    def __call__(self, data: Data) -> Data:
        pass