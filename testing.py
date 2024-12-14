import pybernetics as pb

dataset = pb.Datasets.spiral_data(100, 3)

dense1 = pb.Layers.Dense(2, 3)
activation1 = pb.Layers.ActivationFunction("relu")
dense2 = pb.Layers.Dense(3, 3)
activation2 = pb.Layers.ActivationFunction("leaky relu")
dense3 = pb.Layers.Dense(3, 3)
activation3 = pb.Layers.ActivationFunction("softmax")

neural_network = [dense1, activation1, dense2, activation2, dense3, activation3]

optimizer = pb.Optimizers.StochasticGradientDescent(0.01)

loss_function = pb.Loss.CategoricalCrossentropy()

training_loop = pb.Training.Loop(optimizer, dataset, loss_function, neural_network, 2000)