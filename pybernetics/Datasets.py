from sklearn.datasets import fetch_openml # Sklearn import solely for 'fetch_openml' in the 'fetch' function
import numpy as np

def spiral_data(samples: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    # Copyright (c) 2015 Andrej Karpathy
    # License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
    # Source: https://cs231n.github.io/neural-networks-case-study/

    # Note, only this function for the creation of some spiral data, is NOT mine:

    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

def fetch(name: str, *, version: float, **kwargs) -> object:
    """
    Fetches a verified dataset from OpenML using the specified dataset name and version.
    This function serves as a wrapper around the OpenML `fetch_openml` function to integrate 
    datasets seamlessly into the pybernetics framework.

    Parameters:
        name (str): The name of the dataset to fetch.
        
        version (float): The version of the dataset to fetch. This ensures that the correct 
            version is retrieved.
        
        **kwargs: Additional keyword arguments that are passed directly to the 
            `fetch_openml` function (e.g., `as_frame=True` to return a DataFrame).

    Returns:
        Bunch: The dataset retrieved from OpenML, for more information see fetch_openml.__doc__
        
    Example:
        ```
        dataset = pybernetics.Datasets.fetch("iris", version=1.0, as_frame=True)
        print(dataset)  # Will print the dataset in the requested format.
        ```
    """
    return fetch_openml(name, version=version, **kwargs)