from typing import Callable, Any, Union
from ._Typing import RealNumber
import numpy as np
import math

class Helpers:
    @staticmethod
    def apply_elementwise(func: Callable[[Any], Any], arr: np.ndarray) -> np.ndarray:
        """Returns an array with a function being called on every element of the inputted array"""
        
        vectorized_function = np.vectorize(func)
        return vectorized_function(arr)
    
    class DeprecatedWarning:
        def __init__(self, message: str = "This method is depreciated and may be removed in future updates and/or heavily changed") -> None:
            print(f"DeprecationWarning: {message}")

    class NotYetImplementedWarning:
        def __init__(self, message: str = "This method is not yet implemented and may be added in future updates") -> None:
            print(f"NotYetImplementedWarning: {message}")

class Maths:
    __PI = 3.1415926536
    __SELU_ALPHA = 1.6733
    __SELU_LAMBDA = 1.0507
    __GELU_RECIPROCAL_OF_THE_SQUARE_ROOT_OF_2_PI = 0.79788456
    __GELU_CUBIC_TERM_COEFFICIENT = 0.044715

    @property
    def pi(cls) -> float:
        return cls.__PI
        
    @property
    def selu_alpha(cls) -> float:
        return cls.__SELU_ALPHA

    @property
    def selu_lambda(cls) -> float:
        return cls.__SELU_LAMBDA
    
    @property
    def gelu_reciprocal_of_the_square_root_of_2_pi(cls) -> float:
        return cls.__GELU_RECIPROCAL_OF_THE_SQUARE_ROOT_OF_2_PI
    
    @property
    def gelu_cubic_term_coefficient(cls) -> float:
        return cls.__GELU_CUBIC_TERM_COEFFICIENT

    class ActivationFunctions:
        @staticmethod
        def sigmoid(
                input: RealNumber,
                in_clip_min: RealNumber = -500,
                in_clip_max: RealNumber = 500,
                out_clip_min: RealNumber = 1e-7,
                out_clip_max: RealNumber = 1 - 1e-7,
            ) -> float:
                
            """
            Sigmoid activation function with input and output clipping to prevent overflow.

            Parameters:
            - x: Input value (int or float).
            - in_clip_min: Minimum input value after clipping (default: -500).
            - in_clip_max: Maximum input value after clipping (default: 500).
            - out_clip_min: Minimum output value after clipping (default: 1e-7).
            - out_clip_max: Maximum output value after clipping (default: 1 - 1e-7).

            Returns:
            - The sigmoid of the input after clipping.
            """
                
            # Sigmoid activation function
            # Clip inputs to prevent overflow
            clipped_input = np.clip(input, in_clip_min, in_clip_max)
            # Sigmoid activation function: 1 / (1 + exp(-x))
            output = 1 / (1 + np.exp(-clipped_input))
            # Clip output to avoid extreme values
            return np.clip(output, out_clip_min, out_clip_max)
            
        @staticmethod
        def relu(input: RealNumber) -> float:
            """ReLU activation function. Returns the input if positive, else returns 0."""
            return float(max(0, input))
            
        @staticmethod
        def leaky_relu(input: RealNumber) -> float:
            """Leaky ReLU activation function. Returns a small negative value for negative input, else the input itself."""
            if input < 0:
                return 0.1 * input
            return float(input)

        @staticmethod
        def swish(input: RealNumber) -> float:
            return input * Maths.ActivationFunctions.sigmoid(input)
            
        @staticmethod
        def softmax(inputs: np.ndarray) -> np.ndarray:
            # Subtract the max for numerical stability
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)

        @staticmethod
        def tanh(input: RealNumber) -> float:
            return np.tanh(input)

        @staticmethod
        def elu(input: RealNumber, alpha: RealNumber = 1) -> float:
            if input <= 0:
                return alpha * (np.exp(input) - 1)
            
            return float(input)

        @staticmethod
        def linear(input: RealNumber) -> float:
            return float(input)

        @staticmethod
        def selu(input: RealNumber) -> float:
            if input > 0:
                return Maths.selu_lambda * input
            
            return Maths.selu_lambda * Maths.selu_lambda * (np.exp(input) - 1)
            
        @staticmethod
        def softplus(input: RealNumber) -> float:
            return np.log(1 + np.exp(input))

        @staticmethod
        def softsign(input: RealNumber) -> float:
            return input / (1 + abs(input))

        @staticmethod
        def gelu(input: RealNumber) -> float:
            return 0.5 * input * (1 + np.tanh(np.sqrt(2 / Maths.pi) * (input + 0.044715 * np.power(input, 3))))

        @staticmethod
        def hard_sigmoid(input: RealNumber) -> float:
            return np.clip(0.2 * input + 0.5, 0, 1)
            
        @staticmethod
        def hard_swish(input: RealNumber) -> float:
            return input * Maths.ActivationFunctions.hard_sigmoid(input)

        @staticmethod
        def mish(input: RealNumber) -> float:
            return input * np.tanh(Maths.ActivationFunctions.softplus(input))

        @staticmethod
        def arctan(input: RealNumber) -> float:
            return np.arctan(input)

        @staticmethod
        def signum(input: RealNumber) -> float:
            return np.sign(input)

        @staticmethod
        def binary(input: RealNumber) -> float:
            return 1.0 if input >= 0 else 0.0

        @staticmethod
        def log_sigmoid(input: RealNumber, stability_mode: bool = True) -> float:
            if stability_mode:
                return -np.maximum(0, input) - np.log(1 + np.exp(-np.abs(input)))
            return -np.log(1 + np.exp(-input))

        @staticmethod
        def hardmax(input: Union[np.ndarray, list[RealNumber]]) -> Union[np.ndarray, list]:
            if isinstance(input, np.ndarray):
                output = np.zeros_like(input)
                output[np.argmax(input)] = 1
                return output
            
            elif isinstance(input, list):
                output = [0] * len(input)
                max_index = input.index(max(input))  # Find the index of the maximum value
                output[max_index] = 1  # Set the corresponding index to 1
                return output
            
            else:
                raise ValueError("'Hardmax' expects either a 'numpy.ndarray' or a list of integers (int) or floats (float)")

        class Derivatives:
            @staticmethod
            def sigmoid(
                    input: RealNumber,
                    in_clip_min: RealNumber = -500,
                    in_clip_max: RealNumber = 500,
                    out_clip_min: RealNumber = 1e-7,
                    out_clip_max: RealNumber = 1 - 1e-7
                ) -> float:
                """Derivative of the sigmoid function, respecting clipping boundaries."""
                    
                # Step 1: Calculate the sigmoid of the input (with clipping)
                sigmoid_output = Maths.ActivationFunctions.sigmoid(input, in_clip_min, in_clip_max, out_clip_min, out_clip_max)
                    
                # Step 2: Apply the derivative formula: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                return sigmoid_output * (1 - sigmoid_output)

            @staticmethod
            def relu(input: RealNumber) -> float:
                return 1.0 if input > 0 else 0.0

            @staticmethod
            def leaky_relu(input: RealNumber, alpha: float = 0.1) -> float:
                if input > 0: return 1
                else: return alpha

            @staticmethod
            def swish(input: float) -> float:
                sigmoid_input = Maths.ActivationFunctions.Derivatives.sigmoid(input)  # Use the existing sigmoid function
                return sigmoid_input + input * sigmoid_input * (1 - sigmoid_input)

            @staticmethod
            def softmax(inputs: np.ndarray) -> np.ndarray:
                # Needs work, not fully functional
                # Calculate the softmax output
                sm = Maths.ActivationFunctions.softmax(input)

                # Create an empty matrix for the Jacobian (derivatives)
                jacobian_matrix = np.zeros((len(input), len(input)))

                for i in range(len(input)):
                    for j in range(len(input)):
                        if i == j:
                            jacobian_matrix[i][j] = sm[i] * (1 - sm[i])
                        else:
                            jacobian_matrix[i][j] = -sm[i] * sm[j]

                return jacobian_matrix
                
                """
                Backward pass: Calculate the gradient of the loss with respect to the inputs.

                dinputs = np.empty_like(inputs)

                # Iterate over batches
                for index, (single_output, single_dvalues) in enumerate(zip(output, dvalues)):
                    # Flatten output into a diagonal matrix
                    single_output = single_output.reshape(-1, 1)
                    jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                    
                    # Calculate sample-wise gradient
                    dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
                """

            @staticmethod
            def tanh(input: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
                """
                Compute the derivative of the tanh function element-wise for a given input.
                Or, if the input is a single intiger or floating point value, it will return a single float.

                Parameters:
                - input (numpy.ndarray | int | float): The input array for which the derivative is computed.

                Returns:
                - numpy.ndarray | int | float: The derivative of tanh applied element-wise if in a np.ndarray, else the single float value
                """
                
                return 1 - np.tanh(input) ** 2
            
            @staticmethod
            def elu(input: float, alpha: float = 1.0) -> float:
                if input > 0:
                    return 1
                else:
                    return alpha * np.exp(input)

            @staticmethod    
            def linear(input: RealNumber) -> float:
                return 1.0

            @staticmethod
            def selu(input: float) -> float:
                """Compute the derivative of the SELU activation function."""
                return Maths.selu_lambda if input > 0 else Maths.selu_lambda * Maths.selu_alpha * math.exp(input)

            @staticmethod
            def softplus(input: RealNumber) -> float:
                return math.log(1 + math.exp(input))

            @staticmethod
            def softsign(input: RealNumber) -> float:
                return input / (1 + abs(input))

            @staticmethod
            def gelu(input: RealNumber) -> float:
                """
                Compute the approximate derivative of the GELU activation function.

                Args:
                    input (int | float): The input value.

                Returns:
                    float: The approximate derivative of GELU at the input.
                """
                sqrt_2_over_pi = math.sqrt(2 / Maths.pi)
                h = sqrt_2_over_pi * (input + Maths.gelu_cubic_term_coefficient * input**3)
                tanh_h = math.tanh(h)

                # First term
                term1 = 0.5 * (1 + tanh_h)

                # Second term
                sech_squared = 1 / math.cosh(h)**2
                term2 = 0.5 * input * sech_squared * sqrt_2_over_pi * (1 + 3 * Maths.gelu_cubic_term_coefficient * input**2)

                return term1 + term2

            @staticmethod
            def hard_sigmoid(input: RealNumber) -> float:
                if -1 <= input <= 1:
                    return 0.5
                else:
                    return 0.0

            @staticmethod    
            def hard_swish(input: RealNumber) -> float:
                if -3 <= input <= 3:
                    return (input + 3) / 6 + input / 6
                else:
                    return 0.0

            @staticmethod    
            def mish(input: RealNumber) -> float:
                exp_x = np.exp(input)
                log_term = np.log(1 + exp_x)
                tanh_term = np.tanh(log_term)
                
                # Derivative calculation
                term_1 = tanh_term
                term_2 = (1 - tanh_term ** 2) * (exp_x / (1 + exp_x)) * input
                
                return term_1 + term_2

            @staticmethod
            def signum(input: Union[Any, int, float, complex]) -> float:
                """
                Derivative of the 'Signum' function, though x is technically undifferentiable at x = 0, 0.0 is still returned for simplicity,
                and per standard practise with non-differntiable points along a function
                """ 
                return 0.0 # Hardcoded return of 0 (0.0 as float)

            @staticmethod
            def binary(input: RealNumber) -> float:
                """
                Derivative of the 'Binary' function, though x is technically undifferentiable at x = 0, 0.0 is still returned for simplicity,
                and per standard practise with non-differntiable points along a function
                """ 
                return 0.0

            @staticmethod
            def log_sigmoid(input: RealNumber) -> float:
                sigmoid_input = 1 / (1 + math.exp(-input))
                return 1 - sigmoid_input

            @staticmethod
            def arctan(input: RealNumber) -> float:
                return 1 / (1 + input ** 2)

            def hardmax(input: Union[np.ndarray, list[RealNumber]]) -> Union[np.ndarray, list]:
                if isinstance(input, np.ndarray):
                    output = np.zeros_like(input)  # Initialize as all zeros
                    output[np.argmax(input)] = 1  # Set 1 at the max index
                    return output

                elif isinstance(input, list):
                    output = [0] * len(input)  # Initialize as all zeros
                    output[input.index(max(input))] = 1  # Set 1 at the max index
                    return output

                else:
                    raise ValueError("'hardmax' expects either a 'numpy.ndarray' or a list of integers (int) or floats (float)")
