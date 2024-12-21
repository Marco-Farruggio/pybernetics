"""
Models
=====
Parent model for all neural network typical architectre models, such as:

'Sequential':
    - A simple feedforward neural network model.
    - Equivalent to 'FeedForward', 'Normal', 'Default', 'Linear', 'FullyConnected', 'Dense'.
    - Contains a list of layers, an optimizer, and a loss function.
    - Can be used to train a neural network on a dataset.

    Example:
    ```
    import pybernetics as pb

    sgd_optimizer = pb.Optimizers.SGD(0.01) # Alias for 'stochastic gradient descent' is allowed (as 'SGD')
    cc_loss = pb.Loss.CatergoricalCrossentropy()
    sd_dataset = pb.Datasets.spiral_data(100, 3)

    pbnn = pb.Models.Sequential([
        pb.Layers.Dense(2, 3, "random"),
        pb.Layers.ActivationFunction("sigmoid"),
        pb.Layers.Dense(3, 3, "random")],
        optimizer = sgd_optimizer,
        loss_function = cc_loss)

    pbnn.fit(sd_dataset, 100, alert_freq=None) # Start the training loop
    ```
"""

from . import Training
from ._Typing import Layer, Optimizer, LossFunction, Dataset

class Sequential:
    def __init__(
            self,
            layers: Layer = None,
            optimizer: Optimizer = None,
            loss_function: LossFunction = None
        ) -> None:

        self._model_layers = []
        self._model_optimizer = optimizer
        self._model_loss_function = loss_function

        if layers:
            for layer in layers:
                self.add(layer)

    def add(
            self,
            layer: Layer
        ) -> None:

        self._model_layers.append(layer)

    def fit(
            self,
            dataset: Dataset = None,
            epochs: int = 1000,
            alert: bool = True,
            alert_freq: int = 100,
            debug: bool = False
        ) -> None:
        
        self._model_dataset = dataset
        Training.Loop(optimizer = self._model_optimizer,
                      dataset = self._model_dataset,
                      loss_function = self._model_loss_function,
                      layers = self._model_layers,
                      epochs = epochs,
                      alert = alert,
                      alert_freq = alert_freq,
                      debug = debug)

    def train(self, *args, **kwargs) -> None:
        self.fit(*args, **kwargs)

# NOTE: Ways to save a model coming soon,
# with the 'save' method, and 'load' method.
# Many file file extensions will be supported,
# eg '.csv', '.json', '.h5', '.pkl', etc.

# Allow common aliasing (via class level cloning)
FeedForward = Sequential
Normal = Sequential
Default = Sequential
Linear = Sequential
FullyConnected = Sequential
Dense = Sequential

__all__ = [
    "Sequential",
    "FeedForward",
    "Normal",
    "Default",
    "Linear",
    "FullyConnected",
    "Dense"
]
