"""
Typing
=====

- All custom typings for the pybernetics package, recommended for internal use only.
- Pybernetics keeps with the PEP8 formatting and standard practice for type hinting,
  including useing the builtin module 'typing', and its methods, rather then python's
  builtin datatypes for type hinting (to allow backwards compatability, as using python
  datatypes for hinting is v3.9+

**RealNumber**:
    - The mathematical set of all real numbers.
    - Equivalent to 'Union[int, float]'.
    - Any real number is defined as the following:
        - Positive and negative integers (e.g., -2, 3, 42).
        - Positive and negative floating-point numbers (e.g., -3.14, 2.718, 0.0).
        - Zero (e.g., 0).
        - Rational numbers (e.g., 1/2, 3.5, 7/3).
        - Irrational numbers (e.g., π, √2).
    - This type alias is used to allow both integers and floating-point numbers, ensuring that any numeric value can be handled in contexts where real numbers are required.
    - It is important to note that this does **not** include complex numbers (e.g., `3 + 2j`), as the complex numbers are not considered part of the real number set.
    
    **Usage Example**:
        ```python
        from typing import Union
        
        RealNumber = Union[int, float]
        
        def process_real_number(x: RealNumber) -> RealNumber:
            return x * 2.0  # Example function that processes any real number
        
        # Works with both integer and float values
        result = process_real_number(3)      # Returns 6.0
        result2 = process_real_number(4.5)    # Returns 9.0
        ```

**LossFunction**:
    - A function used to compute the loss in a pybernetics neural network.
    - Equivalent to 'Union[Loss.CategoricalCrossentropy, Loss.MeanSquaredError, Callable[[np.ndarray], np.ndarray]]'.
    - Prefers a pybernetics loss function, from the pybernetics.Loss module.
    - Also excepts custom loss functions, when it recieves a custom loss function, it will do the following:
        - Uses the 'forward', 'backward' and 'compute' methods.
        - Will pass a np.ndarray to 'forward' and expect a np.ndarray in return.
        - Will pass a np.ndarray to 'backward' and not expect anything to be returned, however, it will use the stored value 'dinputs'.
        - 'Compute' is expected to take 2 inputs, the predicted values, and the true values, and will return a float.

**Optimizer**:
    - Any pybernetics optimizer for a neural network built with pybernetics.
    - Equivalent to 'Optimizers.StochasticGradientDescent'.
    - Only one pybernetics optimizer as of yet.
    - Adam tbi and coming soon

**Layer**:
    - A pybernetics layer in a neural network.
    - Can be a neuron layer or an activation function layer (chained seperately).
    - Equivalent to 'Union[Layers.Dense, Layers.ActivationFunction]'.

**Array**:
    - Depreciated.
    - Wrapper type hint for  a 'np.ndarray' (a default numpy matrix of any dimension/magnitute).
    - Equavalent to 'np.ndarray'

**Dataset**:
    - A pybernetics dataset for training.
    - Should be a tuple of (X, y).
    - Each element of the tuple should be a 'np.ndarry'.
    - Equivalent to 'Tuple[np.ndarray, np.ndarray]'.
    - Supports third party datasets, such as 'pybernetics.Datasets.fetch_openml(*args, **kwargs)'.
"""

from typing import Union, Callable, Tuple
from . import Optimizers
from . import Layers
from . import Loss
import numpy as np

RealNumber = Union[int, float]
LossFunction = Union[Loss.CategoricalCrossentropy, Loss.MeanSquaredError, Callable[[np.ndarray], np.ndarray]]
Optimizer = Optimizers.StochasticGradientDescent
Layer = Union[Layers.Dense, Layers.ActivationFunction]
Array = np.ndarray
Dataset = Tuple[np.ndarray, np.ndarray]

__all__ = [
    "RealNumber",
    "LossFunction",
    "Optimizer",
    "Layer",
    "Array", # Deprecated
    "Dataset"
]
