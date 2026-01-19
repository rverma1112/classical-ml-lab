import numpy as np
from .base_optimizer import BaseOptimizer


class AdaGrad(BaseOptimizer):
    def __init__(self, learning_rate, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}

    def step(self, params, grads):
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])

            self.cache[key] += grads[key] ** 2
            params[key] -= (
                self.learning_rate
                * grads[key]
                / (np.sqrt(self.cache[key]) + self.epsilon)
            )
