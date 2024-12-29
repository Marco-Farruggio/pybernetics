"""
Module for handling layers within a nn
"""

import numpy as np
from . import _random_seed
from . import _Utils
from typing import Literal

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
