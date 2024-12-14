Pybernetics
=====

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

- **TrainingDatasets**: Generates or fetches datasets for training, including synthetic datasets 
  like spirals or real-world datasets using OpenML.

- **NaturalLanguageProcessing**: A collection of NLP tools including tokenizers, Markov chains, 
  bag-of-words representations, and character/word predictors.

- **NeuralNetwork**: Implements core neural network functionality:
  - `LayerDense`: Fully connected layers for linear transformations.
  - `ActivationFunction`: Handles activation layers with support for various functions.
  - `Loss`: Base class for loss computation with a concrete implementation for categorical cross-entropy.
  - `OptimizerSGD`: Stochastic Gradient Descent optimizer with support for gradient clipping.
  - `TrainingLoop`: Manages forward and backward passes, loss computation, and weight updates.

Dependencies:
-------------
- **NumPy**: Core dependency for numerical computations.
- **scikit-learn**: Used solely for dataset retrieval via `fetch_openml`.

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
```python
import pybernetics as pb

# Create dataset
X, y = pb.Datasets.spiral_data(100, 3)

# Define network
dense1 = pb.Layers.Dense(2, 3)
activation1 = pb.Layers.ActivationFunction("relu")
dense2 = pb.Layers.Dense(3, 3)
activation2 = pb.Layers.ActivationFunction("softmax")
nn_layers = [dense1, activation1, dense2, activation2]

# Train network
sgd_optimizer = NeuralNetwork.Optimizers.StochasticGradientDescent(0.01)
cc_loss = NeuralNetwork.LossCategoricalCrossentropy()
dataset = X, y

training_loop = NeuralNetwork.TrainingLoop(sgd_optimizer, dataset, cc_loss, nn_layers, 2000)
```

For full documentation and examples, refer to the class-level docstrings or future project documentation.
