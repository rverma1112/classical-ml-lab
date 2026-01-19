from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Base class for all optimization algorithms.
    Optimizers update model parameters using gradients.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, params: dict, grads: dict):
        """
        Perform one optimization step.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters (numpy arrays)
        grads : dict
            Dictionary of gradients (same keys as params)
        """
        pass

    def get_params(self):
        """
        Return optimizer hyperparameters (for UI & logging).
        """
        return {
            "learning_rate": self.learning_rate,
            "optimizer": self.__class__.__name__
        }
