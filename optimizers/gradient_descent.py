from .base_optimizer import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """
    Batch Gradient Descent optimizer.
    """

    def step(self, params, grads):
        for key in params:
            params[key] -= self.learning_rate * grads[key]
