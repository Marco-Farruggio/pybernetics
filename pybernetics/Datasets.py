import numpy as np
from ._Typing import RealNumber
from typing import Union, Literal

Numerical = Union[RealNumber, np.ndarray]

def _convert_to_cartesian_from_polar(radius: Numerical, theta: Numerical) -> tuple[Numerical, Numerical]:
    # Convert polar coordinates to Cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return x, y

def _generate_spiral_arm(num_points: int, p: float, length: float = 2 * np.pi, phase: float = 0, dir_: Literal["cw", "acw"] = "cw", noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    # Define the angle range for the spiral (theta)
    if dir_.lower().strip() == "cw":
        theta = np.linspace(0, -length, num_points)
    
    elif dir_.lower().strip() == "acw":
        theta = np.linspace(0, length, num_points)

    # Define the radius as a function of theta (spiral shape)
    radius = theta * (p + 1) # p controls the tightness of the spiral

    noise_arr = np.random.uniform(-noise, noise, size=radius.shape)
    radius += noise_arr

    # Convert to Cartesian coordinates
    x, y = _convert_to_cartesian_from_polar(radius, theta + phase)
    
    return x, y

def spiral_data(samples: int = 100,
                 classes: int = 3,
                 p: float = 0.9,
                 noise: float = 0.2,
                 one_hot: bool = True,
                 dir_: Literal["cw", "acw"] = "cw",
                 arm_length: float = 2 * np.pi
    ) -> tuple[np.ndarray, np.ndarray]:

    samples_per_class = samples // classes

    X = np.zeros((samples_per_class * classes, 2))  # Initialize the X data
    y = []

    for class_number in range(classes):
        ix = range(samples_per_class * class_number, samples_per_class * (class_number + 1))

        # Generate each spiral arm using the generate_spiral_arm function
        phase = (2 * np.pi / classes) * class_number  # phase shift for each arm
        arm_x, arm_y = _generate_spiral_arm(samples_per_class, p=p, length=arm_length, phase=phase, dir_=dir_, noise=noise)

        # Add the generated spiral arm data to X and the corresponding labels to y
        X[ix] = np.column_stack((arm_x, arm_y))

        if one_hot:
            one_hot_label = [0 for _ in range(classes)]
            one_hot_label[class_number] = 1

            for _ in range(0, samples_per_class):
                y.append(one_hot_label)

        else:
            for _ in range(0, samples_per_class):
                y.append(class_number)

    y = np.array(y)
    return X, y

def sin(x_min: RealNumber = -10, x_max: RealNumber = 10, nsteps: int = 100):
    X = np.linspace(x_min, x_max, nsteps)
    y = np.sin(X)

    return X, y
