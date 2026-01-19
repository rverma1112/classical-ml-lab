import numpy as np
from .base_optimizer import BaseOptimizer


class Momentum(BaseOptimizer):
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocity = {}

    def step(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            self.velocity[key] = (
                self.beta * self.velocity[key]
                + (1 - self.beta) * grads[key]
            )
            params[key] -= self.learning_rate * self.velocity[key]

    def get_params(self):
        base = super().get_params()
        base["beta"] = self.beta
        return base
