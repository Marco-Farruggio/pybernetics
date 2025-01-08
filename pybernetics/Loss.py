import numpy as np

class _BaseLoss:
    def compute(self, outputs: np.ndarray, y_true: np.ndarray) -> float:
        sample_losses = self.forward(outputs, y_true)
        return np.mean(sample_losses)
    
    def __call__(self, outputs: np.ndarray, y_true: np.ndarray) -> float:
        """
        Effectively wraps 'compute' in the call hierarchy to abstract away the call
        """
        return self.compute(outputs, y_true)

class CategoricalCrossentropy(_BaseLoss):
    def __init__(self) -> None:
        pass

    def get_config(self) -> dict:
        return {}
    
    @classmethod
    def from_config(cls, config=None) -> 'CategoricalCrossentropy':
        return cls()

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
    def __init__(self) -> None:
        pass

    def get_config(self) -> dict:
        return {}
    
    @classmethod
    def from_config(cls, config=None) -> 'MeanSquaredError':
        return cls()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the Mean Squared Error (MSE) loss for each sample.
        """
        self.difference = y_pred - y_true
        return self.difference ** 2

    def backward(self) -> None:
        """
        Computes the gradient of the MSE loss with respect to the inputs.
        """
        self.dinputs = 2 * self.difference / len(self.difference)
        return self.dinputs
    
class MeanAbsoluteError:
    def __init__(self) -> None:
        pass

    def forward(self):
        pass

    def backward(self):
        pass

class BinaryCrossentropy:
    def __init__(self) -> None:
        pass

    def forward(self):
        pass

    def backward(self):
        pass

MSE = MeanSquaredError # Allow common aliasing for 'MeanSquaredError'
CC = CategoricalCrossentropy # Allow aliasing for 'CatagoricalCrossentropy'
