from typing import Union, Tuple, Callable

class PyArray:
    def __init__(self, shape: Tuple[int, ...], fill: Union[int, float, str, bool, Callable[[Tuple[int, ...]], Union[int, float, str, bool]]] = 0) -> None:
        self.shape = shape
        self.fill = fill
        self.dims = len(shape)
        self.array = self._create_array(self.shape)

    def _create_array(self, shape, index_tuple=()):
        # Base case: if shape has only one dimension
        if len(shape) == 1:
            return [self._get_value(index_tuple + (i,)) for i in range(shape[0])]
        
        # Recursive case: generate an array for each dimension
        arr = []
        for i in range(shape[0]):  # Create the first dimension
            arr.append(self._create_array(shape[1:], index_tuple + (i,)))  # Recursively create the remaining dimensions
        return arr

    def _iterate_indices(self, shape=None, current_index=()):
        """Generate all possible index tuples for the array."""
        if shape is None:
            shape = self.shape
        
        if len(shape) == 1:
            for i in range(shape[0]):
                yield current_index + (i,)
        else:
            for i in range(shape[0]):
                yield from self._iterate_indices(shape[1:], current_index + (i,))

    def _get_value(self, index_tuple):
        # If fill is callable, call the function with the index tuple
        if callable(self.fill):
            return self.fill(index_tuple)
        # Otherwise, return the fixed fill value
        return self.fill
    
    def __repr__(self):
        return repr(self.array)

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        self.array[index] = value

    def __str__(self):
        return str(self.array).replace("],", "]\n").replace(",", "")

    def __iter__(self):
        return iter(self.array)

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("Shapes of arrays do not match")
        return PyArray(self.shape, fill=lambda index: self[index] + other[index])

    def mean(self):
        total = 0
        count = 0
        for index in self._iterate_indices():
            total += self[index]
            count += 1
        return total / count

    def __mul__(self, other):
        return PyArray(self.shape, fill=lambda index: self[index] * other)

    def __sub__(self, other):
        if self.shape != other.shape:
            raise ValueError("Shapes of arrays do not match")
        return PyArray(self.shape, fill=lambda index: self[index] - other[index])

    def __truediv__(self, other):
        return PyArray(self.shape, fill=lambda index: self[index] / other)

    def __neg__(self):
        return PyArray(self.shape, fill=lambda index: -self[index])

    def __eq__(self, other):
        if self.shape != other.shape:
            return False
        for index in self._iterate_indices():
            if self[index] != other[index]:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __abs__(self):
        return PyArray(self.shape, fill=lambda index: abs(self[index]))

    def __lt__(self, other):
        if self.shape != other.shape:
            raise ValueError("Shapes of arrays do not match")
        return PyArray(self.shape, fill=lambda index: self[index] < other[index])

    def __le__(self, other):
        if self.shape != other.shape:
            raise ValueError("Shapes of arrays do not match")
        return PyArray(self.shape, fill=lambda index: self[index] <= other[index])

    def __gt__(self, other):
        if self.shape != other.shape:
            raise ValueError("Shapes of arrays do not match")
        return PyArray(self.shape, fill=lambda index: self[index] > other[index])

    def __ge__(self, other):
        if self.shape != other.shape:
            raise ValueError("Shapes of arrays do not match")
        return PyArray(self.shape, fill=lambda index: self[index] >= other[index])

    def __deepcopy__(self, memo):
        return PyArray(self.shape, fill=self.fill)

    def __len__(self):
        return len(self.array)

    def __contains__(self, value):
        for index in self._iterate_indices():
            if self[index] == value:
                return True
        return False

# Identity function to generate identity-like arrays
def identity(shape: Tuple[int, ...]) -> PyArray:
    # Define a function that will generate 1 for diagonal elements, and 0 for all others
    def identity_function(index_tuple: Tuple[int, ...]) -> Union[int, float]:
        # Check if all indices in the tuple are the same (diagonal condition)
        if all(i == index_tuple[0] for i in index_tuple):
            return 1
        return 0
    
    # Create and return the identity-like array with the shape and the identity function
    return PyArray(shape, fill=identity_function)

def zeros(shape: Tuple[int, ...]) -> PyArray:
    return PyArray(shape, fill=0)

def ones(shape: Tuple[int, ...]) -> PyArray:
    return PyArray(shape, fill=1)
