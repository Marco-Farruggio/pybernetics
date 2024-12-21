import numpy as np
from . import Layers
from ._Typing import RealNumber

class StochasticGradientDescent:
    def __init__(self, learning_rate: RealNumber = 0.1, clip_value: RealNumber = 5.0) -> None:
        self.learning_rate = learning_rate
        self.clip_value = clip_value

    def update_params(self, layer: Layers.Dense) -> None:
        layer.dweights = np.clip(layer.dweights, -self.clip_value, self.clip_value)
        layer.biases -= self.learning_rate * layer.dbiases
        layer.weights -= self.learning_rate * layer.dweights

# Aliasing (via class level cloning)
SGD = StochasticGradientDescent
