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
from ._Utils import Helpers
from ._Typing import Layer, Optimizer, LossFunction, Dataset
import numpy as np

DeprecatedWarning = Helpers.DeprecatedWarning
NotYetImplementedWarning = Helpers.NotYetImplementedWarning

class Sequential:
    """
    Sequential
    ==========

    - A feedforward neural network model.
    - Contains a list of layers, an optimizer, and a loss function.
    - Can be used to train a neural network on a dataset.
    - Uses n(um)p(y).ndarray(s) for all data processing.

    Methods:
    --------
    'add':
        - Add a layer to the model.

    'fit'/'train':
        - Train the model on a dataset.

    'save':
        - Save the model to a file.
        - Not yet implemented.
    
    'load':
        - Load the model from a file.
        - Not yet implemented.

    'process':
        - Process an input through the model.
        - Does not train the model.

    Parameters:
    -----------
    'layers': List[Layer]
        - The layers of the model.

    'optimizer': Optimizer
        - The optimizer to use when updating the weights of the model.
    
    'loss_function': LossFunction
        - The loss function to use when computing the loss of the model.
    
    Example:
    --------
    ```
    import pybernetics as pb

    sgd_optimizer = pb.Optimizers.SGD(0.01)
    cc_loss = pb.Loss.CC()
    sd_dataset = pb.Datasets.spiral_data(100, 3)

    pbnn = pb.Models.Sequential([
        pb.Layers.Dense(2, 3, "random"),
        pb.Layers.ActivationFunction("sigmoid"),
        pb.Layers.Dense(3, 3, "random"),
        pb.Layers.ActivationFunction("softmax")],
        optimizer = sgd_optimizer,
        loss_function = cc_loss)

    pbnn.fit(sd_dataset, 1000)
    ```
    """
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
            alert_freq: int = None,
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

    def save(self, filename: str = "PyberneticsSequentialModel.csv") -> None:
        """
        Save the model to a file.
        Puts the model into a non volatile storage.
        New model is usually in a non readable format.
        Support for multiple file extension to be implemented.
        """
        NotYetImplementedWarning("Saving a model is not yet implemented.")

    def load(self, filename: str = "PyberneticsSequentialModel.csv") -> None:
        NotYetImplementedWarning("Loading a model is not yet implemented.")

    def __repr__(self) -> str:
        return f"<Sequential Model: {len(self._model_layers)} layers>"

    def __str__(self) -> str:
        return f"Sequential Model: {len(self._model_layers)} layers"

    def __len__(self) -> int:
        return len(self._model_layers)

    def __getitem__(self, index: int) -> Layer:
        return self._model_layers[index]

    def __setitem__(self, index: int, layer: Layer) -> None:
        self._model_layers[index] = layer

    def __delitem__(self, index: int) -> None:
        del self._model_layers[index]

    def __iter__(self):
        return iter(self._model_layers)
    
    def __reversed__(self):
        return reversed(self._model_layers)
    
    def __contains__(self, layer: Layer) -> bool:
        return layer in self._model_layers

    def process(self, input: np.ndarray) -> np.ndarray:
        for layer in self._model_layers:
            layer.forward(input)
            input = layer.output
        
        return input
    
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.process(input)

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
