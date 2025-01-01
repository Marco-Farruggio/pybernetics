"""
Module for handling layers within a nn
"""

import numpy as np
from . import _random_seed
from . import _Utils
from typing import Callable, Literal, Union

RealNumber = Union[int, float]

# Set the seed for reproducibility. NumPy uses a pseudo-random number generator 
# to produce numbers based on an algorithm (like the Box-Muller transform) for Gaussian distribution.
# Mersenne Twister based on seed below
np.random.seed(_random_seed)

class Dense:
    """
    A dense (fully connected) layer in a neural network.

    This layer connects every input to every output with learnable weights and biases. 
    It performs a linear transformation on the input data.

    Attributes:
        weights (numpy.ndarray): The weights of the layer, initialized randomly using np.random.
        biases (numpy.ndarray): The biases of the layer, initialized to zero.

    Methods:
        forward(inputs):
            Computes the output of the layer given the input data.

        backward(dvalues):
            Computes the gradient of the loss with respect to the inputs, weights, and biases.
    """

    def __init__(self, n_inputs: int, n_neurons: int, weights_init: Literal["random", "xavier", "he", "lecun", "zero"] = "random") -> None:
        """
        Initializes an instance of LayerDense

        Parameters:
            n_inputs (int): The number of input features.
            n_neurons (int): The number of neurons (outputs) in the layer.
        """

        # Initialize weights based on the specified method
        if weights_init == "random":
            self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)

        elif weights_init == "xavier":
            # Xavier initialization (Glorot)
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / (n_inputs + n_neurons))

        elif weights_init == "he":
            # He initialization (for ReLU)
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)

        elif weights_init == "lecun":
            # LeCun initialization (for SELU)
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)

        elif weights_init == "zero":
            # All weights initialized to zero
            self.weights = np.zeros((n_inputs, n_neurons))

        else:
            raise ValueError(f"Invalid initialization method: {weights_init}")

        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> None:
        """
        Performs the forward pass of the layer.

        Parameters:
            inputs (numpy.ndarray): The input data to the layer.

        Returns:
            None: The output is stored in the `self.output` attribute.
        """

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Performs the backward pass of the layer.

        Parameters:
            dvalues (numpy.ndarray): The gradient of the loss with respect to the output.

        Returns:
            None: The gradients of the loss with respect to the inputs, weights, and biases 
            are stored in the attributes `self.dinputs`, `self.dweights`, and `self.dbiases`.
        """

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationFunction:
    """
Class handling activation function layers, supported activation functions include:

- Definitions are pythonic (not mathematical, but serve the same purpose) utilising both 'NumPy' and 'math' for calculations and linear algebra
- See pybernetics.NeuralNetwork.ActivationFunction._ACTIVATION_FUNCTIONS and related comments to view all activation functions
- To view the pythonic (code based) definitions for all activation functions and derivaties, see pybernetics._Utils
    """

    # Activation functions in the format:
    #
    # "[reccomended name]": "[reccomended name]",
    # "[common alias]": "[reccomended name]",
    # ...
    #
    # Internals names for multi-word function names are in the format word_word, which is handled by __init__

    # Define activation functions as a class-level constant
    _ACTIVATION_FUNCTIONS = {
        "sigmoid": "sigmoid",
        "logistic": "sigmoid",
        "logistic sigmoid": "sigmoid",
        "logistic_sigmoid": "sigmoid",
        "sig": "sigmoid",
        "sigma": "sigmoid",

        "binary": "binary",
        "step": "binary",
        "binary step": "binary",
        "binary_step": "binary",
        "bool": "binary",
        "boolean": "binary",

        "tanh": "tanh",
        "hyperbolic tangent": "tanh",

        "relu": "relu",
        "rectified linear unit": "relu",
        "rectifier": "relu",

        "leaky relu": "leaky relu",
        "leaky rectified linear unit": "leaky relu",
        "leaky_relu": "leaky relu",

        "softmax": "softmax",
                
        "swish": "swish",
        "silu": "swish",

        "elu": "elu",
        "exponential linear unit": "elu",

        "gelu": "gelu",
        "gaussian error linear unit": "gelu",

        "selu": "selu",
        "scaled exponential linear unit": "selu",
        "scaled elu": "selu",

        "linear": "linear",
        "identity": "linear",
        "none": "linear",

        "softplus": "softplus",

        "softsign": "softsign",

        "hard sigmoid": "hard sigmoid",
        "hard_sigmoid": "hard sigmoid",

        "hard swish": "hard swish",
        "hard_swish": "hard swish",

        "mish": "mish",
                
        "arctan": "arctan",
        "arctangent": "arctan",
        "atan": "arctan",

        "signum": "signum",
        "sign": "signum",

        "logsigmoid": "logsigmoid",
        "logsig": "logsigmoid",
        "log sigmoid": "logsigmoid",
        "log sig": "logsigmoid",

        "hardmax": "hardmax",
        "hard max": "hardmax",
        "hard_max": "hardmax"
    }

    def __init__(self, function: str) -> None:
        # Normalize the function name using the dictionary
        normalized_function  = self._ACTIVATION_FUNCTIONS.get(function.lower().strip())
            
        if normalized_function is None:
            supported_functions = ", ".join(sorted(set(self._ACTIVATION_FUNCTIONS.values())))
            raise NotImplementedError(f"Activation function '{function}' is not implemented.\nSupported functions are: {supported_functions}")

        self.function = normalized_function

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs

        if self.function == "sigmoid":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.sigmoid, self.inputs)

        elif self.function == "relu":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.relu, self.inputs)

        elif self.function == "tanh":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.tanh, self.inputs)

        elif self.function == "binary":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.binary, self.inputs)

        elif self.function == "leaky relu":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.leaky_relu, self.inputs)

        elif self.function == "elu":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.elu, self.inputs)

        elif self.function == "softplus":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.softplus, self.inputs)

        elif self.function == "selu":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.selu, self.inputs)

        elif self.function == "gelu":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.gelu, self.inputs)

        elif self.function == "logsigmoid":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.log_sigmoid, self.inputs)

        elif self.function == "swish":
            self.output = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.swish, self.inputs)

        elif self.function == "hardmax":
            self.output = _Utils.Maths.ActivationFunctions.hardmax(self.inputs)
        
        elif self.function == "softmax":
            self.output = _Utils.Maths.ActivationFunctions.softmax(self.inputs)
    
    def backward(self, dvalues: np.ndarray) -> None:
        if self.function == "sigmoid":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.sigmoid, self.inputs)

        elif self.function == "relu":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.relu, self.inputs)

        elif self.function == "tanh":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.tanh, self.inputs)

        elif self.function == "binary":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.binary, self.inputs)

        elif self.function == "leaky relu":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.leaky_relu, self.inputs)

        elif self.function == "swish":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.swish, self.inputs)

        elif self.function == "elu":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.elu, self.inputs)

        elif self.function == "linear":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.linear, self.inputs)

        elif self.function == "softplus":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.softplus, self.inputs)

        elif self.function == "softsign":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.softsign, self.inputs)

        elif self.function == "hard sigmoid":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.hard_sigmoid, self.inputs)

        elif self.function == "hard swish":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.hard_swish, self.inputs)

        elif self.function == "mish":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.mish, self.inputs)

        elif self.function == "arctan":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.arctan, self.inputs)

        elif self.function == "signum":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.signum, self.inputs)

        elif self.function == "log sigmoid":
            self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.log_sigmoid, self.inputs)
 
        elif self.function == "softmax":
            # self.dinputs = dvalues * _Utils.Maths.ActivationFunctions.Derivatives.softmax(self.inputs)
            
            # Reshape output to a column vector
            # softmax_output = self.output.reshape(-1, 1)
            # Compute the Jacobian matrix
            # jacobian_matrix = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)
            # Multiply the Jacobian matrix by the gradient from the next layer
            # self.dinputs = np.dot(jacobian_matrix, dvalues)

            # Initialize the array for storing the gradient w.r.t. inputs
            self.dinputs = np.zeros_like(self.inputs)
            
            # Iterate over each sample in the batch
            for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
                # Flatten the output
                single_output = single_output.reshape(-1, 1)
                # Compute the Jacobian matrix for the sample
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                # Compute the gradient for this sample
                self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        elif self.function == "hardmax":
            self.dinputs = dvalues * _Utils.Maths.ActivationFunctions.Derivatives.hardmax(self.inputs)

class Sigmoid:
    def __init__(self, in_clip_min: RealNumber = -500, in_clip_max: RealNumber = 500, out_clip_min: RealNumber = 1e-7, out_clip_max: RealNumber = 1 - 1e-7) -> None:
        self.in_clip_min = in_clip_min
        self.in_clip_max = in_clip_max
        self.out_clip_min = out_clip_min
        self.out_clip_max = out_clip_max
        
    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.sigmoid, self.inputs, self.in_clip_min, self.in_clip_max, self.out_clip_min, self.out_clip_max)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.sigmoid, self.inputs, self.in_clip_min, self.in_clip_max, self.out_clip_min, self.out_clip_max)

class ReLU:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.relu, self.inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.relu, self.inputs)

class Tanh:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.tanh, self.inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.tanh, self.inputs)

class Binary:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.binary, self.inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.binary, self.inputs)

class LeakyReLU:
    def __init__(self, alpha: RealNumber) -> None: # NOTE: UPDATE REALNUMBER IMPORT FROM _TYPING
        self.alpha = alpha

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.leaky_relu, self.inputs, self.alpha)

    def backward(self, dvalues: np.ndarray) -> None:
        self.outputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.leaky_relu, self.inputs, self.alpha)

class Swish:
    def __init__(self, beta: RealNumber = 1) -> None: # NOTE: UPDATE REALNUMBER IMPORT FROM _TYPING
        self.beta = beta

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = _Utils.Maths.ActivationFunctions.swish(inputs, self.beta)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * _Utils.Helpers.apply_elementwise(_Utils.Maths.ActivationFunctions.Derivatives.swish, self.inputs) # NEEDS WORK

class ELU:
    def __init__(self, alpha: RealNumber = 1) -> None:
        self.alpha = alpha

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = _Utils.Maths.ActivationFunctions.elu(inputs, self.alpha)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * _Utils.Maths.ActivationFunctions.Derivatives.elu(self.inputs, self.alpha)

class Softmax:
    def __init__(self) -> None:
        pass
    
    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))  # Prevent overflow
        self.outputs = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    
    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = np.zeros_like(dvalues)

        for i in range(len(dvalues)):
            jacobian_matrix = np.diagflat(self.outputs[i]) - np.outer(self.outputs[i], self.outputs[i])
            self.dinputs[i] = np.dot(jacobian_matrix, dvalues[i])

class SELU:
    def __init__(self, alpha: RealNumber = 1.67326, scale: RealNumber = 1.0507) -> None:
        self.alpha = alpha  # Coefficient for the negative part of the function
        self.scale = scale  # Scaling factor for the output

    def forward(self, inputs: np.ndarray) -> None:
        """Applies the SELU activation function element-wise on the input array."""
        self.inputs = inputs
        self.outputs = self.scale * np.where(self.inputs > 0, self.inputs, self.alpha * (np.exp(self.inputs) - 1))

    def backward(self, dvalues: np.ndarray) -> None:
        """Calculates the gradient (derivative) of the SELU activation function."""
        self.dinputs = dvalues * self.scale * np.where(self.inputs > 0, 1, self.alpha * np.exp(self.inputs))

class GELU:
    def __init__(self) -> None:
        pass

    @staticmethod
    def gelu(input: np.ndarray) -> np.ndarray:
        # GELU function
        return 0.5 * input * (1 + np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * np.power(input, 3))))

    @staticmethod
    def gelu_derivative(input: np.ndarray) -> np.ndarray:
        # Derivative of GELU
        factor = np.sqrt(2 / np.pi)
        z = factor * (input + 0.044715 * np.power(input, 3))
        sech2 = 1 - np.tanh(z)**2  # sech^2(z)
        return 0.5 * (1 + np.tanh(z)) + input * sech2 * factor * (1 + 0.13345 * np.power(input, 2))

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = self.gelu(self.inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * self.gelu_derivative(self.inputs)

class Softplus:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the Softplus activation."""
        self.inputs = inputs
        self.outputs = np.log(1 + np.exp(inputs))
    
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """Compute the derivative of Softplus with respect to the inputs."""
        self.dinputs = dvalues * (1 / (1 + np.exp(-self.inputs)))

class Arctan:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """Compute the Arctan (inverse tangent) activation."""
        self.inputs = inputs
        self.outputs = np.arctan(inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        """Compute the derivative of Arctan with respect to the inputs."""
        self.dinputs = dvalues / (1 + self.inputs**2)  # Derivative of Arctan

class Signum:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """Compute the Signum (sign) activation."""
        self.inputs = inputs
        self.outputs = np.sign(inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        """Compute the derivative of Signum with respect to the inputs."""
        # The derivative of the signum function is 0 everywhere except at 0,
        # where it's undefined, so we simply return 0 as a placeholder.
        self.dinputs = np.zeros_like(self.inputs)

class Hardmax:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """Compute the Hardmax activation."""
        self.inputs = inputs
        # Find the index of the maximum value in the array
        self.outputs = np.zeros_like(self.inputs)
        self.outputs[np.argmax(self.inputs)] = 1

    def backward(self, dvalues: np.ndarray) -> None:
        """Compute the derivative of Hardmax."""
        # Initialize the gradient with zeros
        self.dinputs = np.zeros_like(self.inputs)

        # Set the gradient to 1 for the index of the maximum value
        self.dinputs[np.argmax(self.inputs)] = 1  # Derivative is 1 at the max value index

class LogSigmoid:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """Compute the LogSigmoid activation."""
        self.inputs = inputs
        self.outputs = -np.log(1 + np.exp(-self.inputs))  # LogSigmoid formula

    def backward(self, dvalues: np.ndarray) -> None:
        """Compute the derivative of LogSigmoid."""
        sigmoid = 1 / (1 + np.exp(-self.inputs))  # Sigmoid function
        self.dinputs = dvalues * sigmoid * (1 - sigmoid)  # Sigmoid derivative

class ReLU6:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """Forward pass for ReLU6."""
        self.inputs = inputs
        self.outputs = np.clip(self.inputs, 0, 6)  # Clipping values between 0 and 6

    def backward(self, dvalues: np.ndarray) -> None:
        """Backward pass for ReLU6."""
        self.dinputs = dvalues * (self.inputs > 0) * (self.inputs < 6)  # Derivative for ReLU6

class Clip:
    def __init__(self, min_: RealNumber, max_: RealNumber) -> None:
        """
        Initialize the Clip activation function.
        :param min_: Minimum value for clipping.
        :param max_: Maximum value for clipping.
        """
        self.clip_min = min_
        self.clip_max = max_

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass for the Clip function.
        :param inputs: Input array to clip.
        """
        self.inputs = inputs
        self.outputs = np.clip(self.inputs, self.clip_min, self.clip_max)

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Backward pass for the Clip function.
        :param dvalues: Gradients flowing from the next layer.
        """
        # Gradients are passed only where the inputs are not clipped
        self.dinputs = dvalues * ((self.inputs > self.clip_min) & (self.inputs < self.clip_max))

class Normalize:
    def __init__(self, in_min: RealNumber, in_max: RealNumber, out_min: RealNumber, out_max: RealNumber) -> None:
        """
        Initialize the Normalization activation function.
        :param in_min: Minimum (inclusive) value of the input.
        :param in_max: Maximum (inclusive) value of the input.
        :param out_min: Minimum (inclusive) value of the output.
        :param out_max: Maximum (inclusive) value of the output.
        """
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass for normalization.
        :param inputs: Input array to normalize.
        """
        self.inputs = inputs
        # Perform normalization: scale inputs to the output range
        self.outputs = self.out_min + (inputs - self.in_min) * (self.out_max - self.out_min) / (self.in_max - self.in_min)

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Backward pass for normalization.
        :param dvalues: Gradients flowing from the next layer.
        """
        # Gradients pass unchanged through normalization, scaled by the input-output ratio
        self.dinputs = dvalues * (self.out_max - self.out_min) / (self.in_max - self.in_min)

class TReLU:
    def __init__(self, theta: RealNumber = 1.0) -> None:
        self.theta = theta

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = np.where(self.inputs > self.theta, self.inputs, 0)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * (self.inputs > self.theta)

class Custom:
    def __init__(self, activation_function: Callable[[np.ndarray], np.ndarray], derivative_function: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Initialize the custom activation function.

        Both the forward and backward pass functions will be passed a np.ndarray when they are called.

        :param activation_function: A callable function for the forward pass (e.g., f(x)).
        :param derivative_function: A callable function for the backward pass (e.g., f'(x)).
        """
        self.activation_function = activation_function
        self.derivative_function = derivative_function

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass for the custom activation function.

        :param inputs: Input array.
        """
        self.inputs = inputs
        self.outputs = self.activation_function(self.inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Backward pass for the custom activation function.

        :param dvalues: Gradients flowing from the next layer.
        """
        self.dinputs = dvalues * self.derivative_function(self.inputs)
