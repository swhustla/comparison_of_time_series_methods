from typing import TypeVar, Protocol

Data = TypeVar("Data", contravariant=True)


class Decompose(Protocol[Data, Data]):
    def __call__(self, data: Data) -> Data:
        pass

