"""
Pybernetics
===========

Pybernetics is a lightweight Python toolkit for developing and training neural networks from scratch. 
It is designed to be a self-contained library, avoiding the use of third-party machine learning 
or deep learning frameworks. Pybernetics relies on NumPy for matrix operations and incorporates 
handcrafted implementations for common neural network components such as layers, activation functions, 
and optimizers.

Key Features:
-------------
- **Lightweight and Modular**: Provides essential tools for building and training neural networks 
  while maintaining simplicity and flexibility.
- **Custom Activation Functions**: Includes a variety of activation functions implemented using NumPy 
  for high performance and easy customization.
- **Dataset Integration**: Offers utilities to generate synthetic datasets or fetch real-world datasets 
  via scikit-learn's `fetch_openml` (used solely for dataset retrieval).
- **Utilities for NLP**: Supports tokenization, bag-of-words, Markov chains, and other natural 
  language processing methods tailored for neural network use cases.

Modules and Classes:
--------------------
- **_Utils**: Internal utility functions for mathematical operations and helper methods, including:
    - `Maths`: Implements activation functions such as ReLU, sigmoid, softmax, and their derivatives.
    - `Helpers`: Provides methods for element-wise operations on NumPy arrays.

- **Dataset**: Generates or fetches datasets for training, including synthetic datasets 
    like spirals or real-world datasets using OpenML.

- **NaturalLanguageProcessing**: A collection of NLP tools including tokenizers, Markov chains, 
    bag-of-words representations, and character/word predictors.

- **Layers**: Contains classes for building neural network layers, including:
    - `Dense`: Fully connected layers with customizable input and output sizes.
    - `Sigmoid`: Implements the sigmoid activation function for neural network layers.
    - `ReLU`: Implements the ReLU activation function for neural network layers.
    - `Tanh`: Implements the tanh activation function for neural network layers.
    - `Binary`: Implements a binary step activation function.
    - `LeakyReLU`: Implements the leaky ReLU activation function with a customizable alpha parameter.
    - `Swish`: Implements the Swish activation function with a customizable beta parameter.
    - `ELU`: Implements the ELU activation function with a customizable alpha parameter.
    - `Softmax`: Implements the softmax activation function for probability distributions.
    - `SELU`: Implements the SELU activation function with alpha and scale parameters.
    - `GELU`: Implements the Gaussian Error Linear Unit activation function.
    - `Softplus`: Implements the softplus activation function.
    - `Arctan`: Implements the arctan activation function.
    - `Signum`: Implements the sign function for activation.
    - `Hardmax`: Implements the hardmax activation function.
    - `LogSigmoid`: Implements the log-sigmoid activation function.
    - `ReLU6`: Implements the ReLU6 activation function with output clipping between 0 and 6.
    - `TReLU`: Implements the thresholded ReLU (TReLU) activation function.
    - `Clip`: Clips inputs to a defined minimum and maximum value range.
    - `Normalize`: Normalizes inputs to a specified range.
    - `Dropout`: Implements the Dropout layer
    - `ZeroCenteredSigmoid`: Custom author designed activation function.
    - `Custom`: Allows defining custom activation functions and their derivatives.
    - `Conv1D`: Implements a sliding 1D kernal applied to the input.

- **Loss**: Defines loss functions for neural network training, including:
    - `CategoricalCrossentropy`: Computes the cross-entropy loss for classification tasks.
    - `MeanSquaredError`: Calculates the mean squared error for regression tasks.

- **Optimizers**: Provides optimization algorithms for training neural networks, including:
    - `SGD`: Stochastic Gradient Descent optimizer with customizable learning rate.

- **Training**: Contains classes for training neural networks, including:
    - `Loop`: The main training loop for training neural networks with specified optimizers, 
        loss functions, and layers.

- **Models**: Defines high-level models for training neural networks, including:
    - `Sequential`: A feedforward neural network model that can be trained on datasets.
    - `load`: Loads a pybernetics saved neural network.

- **_Typing**: (Internal) Type hints for classes and functions, including custom types for neural network components.

- **DataTypes**: Custom data types for neural network components.

Dependencies:
-------------
- **NumPy**: Core dependency for numerical computations.

Built-in modules:
- **typing**: Typing for all classes and functions
- **re**: RegEx used for fast non-pythonic language filtering and substitution
- **collections**: 'Defaultdict' used in NLP

Metadata:
---------
- Author: Marco Farruggio
- License: MIT
- Version: 4.5.3
- Status: Development
- Created: 2024-11-28
- Platform: Cross-platform

Usage:
------
Import pybernetics and utilize its modular components to design, train, and evaluate neural networks 
or integrate its NLP tools into your projects.

Example:
--------
```
import pybernetics as pb
import numpy as np

sgd_optimizer = pb.Optimizers.SGD(0.01)
cc_loss = pb.Loss.CC()
sd_dataset = pb.Datasets.spiral_data(100, 3)

pbnn = pb.Models.Sequential([
    pb.Layers.Dense(2, 3, "random"),
    pb.Layers.Sigmoid(-750, 750),
    pb.Layers.Dense(3, 3, "random"),
    pb.Layers.Tanh(),
    pb.Layers.Dense(3, 3, "random")],
    optimizer = sgd_optimizer,
    loss_function = cc_loss)

pbnn.fit(sd_dataset, 1000)
```

For full documentation and examples, refer to the class-level docstrings or future project documentation.

Importing
---------

It is recommended to import the library using the following syntax:

```
import pybernetics as pb
```

As most of the doc strings do not include the full module path, this will make it easier to understand.
Also it is recommended to use the 'pb' alias as it is shorter and easier to type.
Shortens attribute chaining and makes the code more readable.

Maths
-----
- All mathematical operations are performed using NumPy arrays. As every function expects.
- Memory usage optimized by in place operations and variable reusage and reallocation
- Speed optimized using NumPy for base C computational speed with arrays

Dedication
----------
- Sam Blight"""

# Base dunders & Metadata
__version__ = "0.1.1"
__author__ = "Marco Farruggio"
__maintainer__ = "Marco Farruggio"
__email__ = "marcofarruggiopersonal@gmail.com"
__status__ = "development"
__platform__ = "Cross-platform"
__dependencies__ = ["numpy"]
__created__ = "2024-05-12" # Rough estimate
__license__ = "MIT" # Open-source community
__description__ = "Pybernetics is a lightweight toolkit for the development and training of neural networks."
__github__ = "https://github.com/WateryBird/pybernetics"
__url__ = __github__
_random_seed = 0

from . import _Utils # No Circular Imports
from . import _Typing # Typehinting lazy imports styles to not need dependencies
from . import Datasets # No Circular Imports
from . import Layers # Requires _Utils
from . import NaturalLanguageProcessing # Required __version__
from . import Loss # Requires Layers, __version__
from . import Optimizers # Requires Layers
from . import Training # Requires Optimizer, Layers and __version__
from . import Models # Requires ^^^
from . import PyArrays
from . import DataTypes

__all__ = [
    "Datasets",
    "Layers",
    "NaturalLanguageProcessing",
    "Loss",
    "Optimizers",
    "Training",
    "Models",
    "PyArrays",
    "DataTypes"
]
