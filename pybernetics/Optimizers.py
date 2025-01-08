import numpy as np
from . import Layers
from typing import Union

RealNumber = Union[int, float] # Work around circular import from _Typing.py

class StochasticGradientDescent:
    def __init__(self, learning_rate: RealNumber = 0.1, clip_value: RealNumber = 5.0) -> None:
        self.learning_rate = learning_rate
        self.clip_value = clip_value

    def update_params(self, layer: Layers.Dense) -> None:
        layer.dweights = np.clip(layer.dweights, -self.clip_value, self.clip_value)
        layer.biases -= self.learning_rate * layer.dbiases
        layer.weights -= self.learning_rate * layer.dweights

    def get_config(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "clip_value": self.clip_value
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'StochasticGradientDescent':
        return cls(**config)

class GradientDescent:
    def __init__(self) -> None:
        pass

# Aliasing (via class level cloning)
SGD = StochasticGradientDescent
GD = GradientDescent
