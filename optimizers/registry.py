from .gradient_descent import GradientDescent
from .stochastic_gradient_descent import StochasticGradientDescent
from .mini_batch_gd import MiniBatchGradientDescent
from .momentum import Momentum
from .nesterov import Nesterov
from .adagrad import AdaGrad
from .rmsprop import RMSProp
from .adam import Adam


OPTIMIZER_REGISTRY = {
    "Gradient Descent": GradientDescent,
    "Stochastic GD": StochasticGradientDescent,
    "Mini-Batch GD": MiniBatchGradientDescent,
    "Momentum": Momentum,
    "Nesterov": Nesterov,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
    "Adam": Adam
}
