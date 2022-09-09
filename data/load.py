from typing import Generator, Protocol, TypeVar

Data = TypeVar("Data")

class Load(Protocol[Data]):
    def __call__(self, count: int) -> Generator[Data, None, None]:
        pass