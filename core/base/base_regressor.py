from .base_model import BaseModel

class BaseRegressor(BaseModel):
    """
    Base class for all regression models.
    """

    def predict(self, X):
        """
        Must return continuous predictions.
        """
        raise NotImplementedError
