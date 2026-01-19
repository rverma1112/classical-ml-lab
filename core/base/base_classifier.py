from .base_model import BaseModel

class BaseClassifier(BaseModel):
    """
    Base class for all classification models.
    """

    def predict_proba(self, X):
        """
        Optional: return class probabilities.
        """
        raise NotImplementedError
