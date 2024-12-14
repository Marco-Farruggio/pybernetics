from typing import Tuple, List, Union
from . import __version__
from . import Layers
import numpy as np

class _BaseLoss:
    def compute(self, output: np.ndarray, y: np.ndarray) -> float:
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossentropy(_BaseLoss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        samples = len(y_pred)
        # Clip predictions to prevent division by zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Initialize the gradient
        self.dinputs = np.empty_like(y_pred_clipped)

        if len(y_true.shape) == 1:  # Scalar values
            self.dinputs[range(samples), y_true] = -1 / y_pred_clipped[range(samples), y_true]
        
        elif len(y_true.shape) == 2:  # One-hot encoded vectors
            self.dinputs = -y_true / y_pred_clipped
        
        # Normalize the gradient (for mean loss)
        self.dinputs /= samples

class MeanSquaredError(_BaseLoss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the Mean Squared Error (MSE) loss for each sample.
        """
        return np.mean(np.square(y_pred - y_true), axis=-1)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Computes the gradient of the MSE loss with respect to the inputs.
        """
        samples = len(y_pred)
        # Gradient of MSE: 2 * (y_pred - y_true) / N
        self.dinputs = 2 * (y_pred - y_true) / samples

MSE = MeanSquaredError # Allow common aliasing for 'MeanSquaredError'
CC = CategoricalCrossentropy # Allow aliasing for 'CatagoricalCrossentropy'