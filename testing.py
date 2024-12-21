import pybernetics as pb

"""
# Create dataset
X, y = pb.Datasets.spiral_data(100, 3)

# Define network
dense1 = pb.Layers.Dense(2, 3)
activation1 = pb.Layers.ActivationFunction("relu")
dense2 = pb.Layers.Dense(3, 3)
nn_layers = [dense1, activation1, dense2]

# Train network
sgd_optimizer = pb.Optimizers.StochasticGradientDescent(0.01)
cc_loss = pb.Loss.CategoricalCrossentropy()
dataset = X, y

training_loop = pb.Training.Loop(sgd_optimizer, dataset, cc_loss, nn_layers, 2000)

pb._Random.testrun()

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
"""

sgd_optimizer = pb.Optimizers.SGD(0.01)
cc_loss = pb.Loss.CC()
sd_dataset = pb.Datasets.spiral_data(100, 3)

pbnn = pb.Models.Sequential([
    pb.Layers.Dense(2, 3, "random"),
    pb.Layers.ActivationFunction("sigmoid"),
    pb.Layers.Dense(3, 3, "random")],
    optimizer = sgd_optimizer,
    loss_function = cc_loss)

pbnn.fit(sd_dataset, 100, alert_freq=None)
