from typing import Generator, Protocol, TypeVar

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction", covariant=True)

class Predict(Protocol[Data, Prediction]):
    def __call__(self, data: Data) -> Generator[Prediction, None, None]:
        pass