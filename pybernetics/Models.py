"""
NYI
=====
Parent model for all neural network typical architectre models, such as:
- Sequential (FeedForward)
"""

from . import Layers
from . import Loss
from . import Datasets
from . import Optimizers
from typing import Union, List

class Sequential:
    def __init__(
            self,
            layers: List[Union[Layers.Dense, Layers.ActivationFunction]] = None,
            optimizer: Optimizers.StochasticGradientDescent = None,
            loss: Loss.CategoricalCrossentropy = None
        ) -> None:
        
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss

        raise NotImplementedError("NYI")

    def train(self, dataset: Union[Datasets.fetch, Datasets.spiral_data, List[List]] = None, epochs: int = 1000) -> None:
        raise NotImplementedError("NYI")