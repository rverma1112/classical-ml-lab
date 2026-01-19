from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Base interface for all models in Classical ML Lab.
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
