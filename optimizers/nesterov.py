import numpy as np
from .momentum import Momentum


class Nesterov(Momentum):
    def step(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            v_prev = self.velocity[key]
            self.velocity[key] = (
                self.beta * self.velocity[key]
                + self.learning_rate * grads[key]
            )

            params[key] -= (
                -self.beta * v_prev
                + (1 + self.beta) * self.velocity[key]
            )
