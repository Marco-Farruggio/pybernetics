import numpy as np
from ._Typing import RealNumber

def spiral_data(samples: int = 100, classes: int = 3, noise = 0.2) -> tuple[np.ndarray, np.ndarray]:
    # Copyright (c) 2015 Andrej Karpathy
    # License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
    # Source: https://cs231n.github.io/neural-networks-case-study/

    # Modified the code, as so that you can customize the noise
    
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype="uint8")

    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1.0, samples)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples)
        
        if noise is not None or noise != 0:
            t = t + np.random.randn(samples) * noise

        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number

    return X, y

def sin(x_min: RealNumber = -10, x_max: RealNumber = 10, nsteps: int = 100):
    X = np.linspace(x_min, x_max, nsteps)
    y = np.sin(X)

    return X, y
