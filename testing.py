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

inputs = np.array([[0.1, 0.2]])
print(pbnn.process(inputs))

my_array = pb.PyArrays.PyArray((2, 3, 4), fill=0)
print(my_array)

my_array2 = pb.PyArrays.PyArray((3, 2), fill=0)
print(my_array2)

identity_matrix = pb.PyArrays.identity((3, 3))
print(identity_matrix)

identity_matrix_3d = pb.PyArrays.identity((3, 3, 3))
print(identity_matrix_3d)

for index in identity_matrix._iterate_indices():
    print(f"Index: {index}")
