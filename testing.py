import pybernetics as pb

sgd_optimizer = pb.Optimizers.SGD(0.01)
cc_loss = pb.Loss.CC()
sd_dataset = pb.Datasets.spiral_data(100, 3)

pbnn = pb.Models.Sequential([
    pb.Layers.Dense(2, 3, "random"),
    pb.Layers.Sigmoid(),
    pb.Layers.Dense(3, 3, "random"),
    pb.Layers.Tanh(),
    pb.Layers.Dense(3, 3, "random")],
    optimizer = sgd_optimizer,
    loss_function = cc_loss)

pbnn.fit(sd_dataset, 1000)
