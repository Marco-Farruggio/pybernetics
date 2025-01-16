from typing import List
from . import Layers
from . import __version__
from ._Typing import Optimizer, LossFunction, Layer, Dataset

class Loop:
    """
    Loop
    =====
    The main training llop for any neural network created by pybernetics,
    this class is responsible for training the network using the specified dataset,
    optimizer, loss function, and layers. The class will perform a forward pass
    through the network, compute the loss, perform a backward pass, and update the
    weights of the network using the optimizer. The class will also print a progress
    bar and the loss at each epoch.

    This class is also wrapped by the 'fit'/'train' method in the Sequential model
    class, which is a more user-friendly way to train a neural network.

    Parameters:
    -----------
    
    'optimizer': Optimizer
        - The optimizer to use when updating the weights of the network.
    
    'dataset': Dataset
        - The dataset to train the network on.
        - this should be a tuple of X and y.
    
    'loss_function': LossFunction
        - The loss function to use when computing the loss of the network.

    'layers': List[Layer]
        - The layers of the network to train.
        
    'epochs': int
        - The number of epochs to train the network for.
    
    'alert': bool
        - Whether or not to print the progress sheet at each epoch.
    
    'alert_freq': int
        - The frequency at which to print the progress sheet
        - If None, print at each and every epoch.

    'debug': bool
        - Whether or not to print the output of the network at each epoch.

    Returns:
    --------
    None
    """
    def __init__(
            self,
            optimizer: Optimizer,
            dataset: Dataset,
            loss_function: LossFunction,
            layers: List[Layer],
            epochs: int = 1000,
            alert: bool = True,
            alert_freq: int = None,
            debug: bool = False
        ) -> None:

        # Initialize the training loop
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.layers = layers
        self.X, self.y = dataset  # Unpack the dataset into X and y (paired dataset)
        self.epochs = epochs
        self.alert = alert
        self.alert_freq = alert_freq
        self.debug = debug

        # Progress bar settings
        self.pbar_len = 10
        self.first_loss = None
        self.pbar_full_char = "#"
        self.pbar_empty_char = "-"

        for epoch in range(1, epochs + 1):
            inputs = self.X
            
            if self.debug:
                print(f"Inputs: {inputs}")
            
            # Perform a forward pass through the full network
            for layer in layers:
                layer.forward(inputs)
                inputs = layer.outputs

            # Compute loss
            loss = loss_function.compute(inputs, self.y)
            
            if self.first_loss is None:
                self.first_loss = loss

            # Perform a backward pass
            loss_grad = loss_function.backward()

            reversed_layers = self.layers[::-1]

            reversed_layers[0].backward(loss_grad)

            last_layer = reversed_layers[0]

            for layer in reversed_layers[1:]:
                layer.backward(last_layer.dinputs)
                last_layer = layer

            # Update weights dynamically using gradient descent
            for layer in self.layers:
                if isinstance(layer, Layers.Dense):
                    self.optimizer.update_params(layer=layer)

            if self.alert:
                # Calculate the percentage of completion
                percent = (epoch / epochs) * 100

                # Update loading bar and print the loss
                num_full_chars = int(self.pbar_len * (percent / 100)) # Float - int conversion (via truncation)
                pbar = f"[{self.pbar_full_char * num_full_chars}{self.pbar_empty_char * (self.pbar_len - num_full_chars)}]"

                # Variables formatted nicely for printing
                formatted_epochs = f"{epoch}/{epochs}"
                formatted_percentage = f"{(epoch / epochs) * 100:0<6.2f}%"[:7]
                formatted_loss = f"{loss:.5f}"[:7]
                formatted_total_improvement = f"{(self.first_loss - loss):0<16.5f}"[:7]
                epochs_formatting_length = len(str(epochs))

                if self.alert_freq is not None:
                    # If the alert frequency is set, only print the progress sheet at those intervals
                    if epoch % self.alert_freq == 0 or epoch == self.epochs:
                        # Print the progress sheet
                        print(f"Training: {pbar} {formatted_percentage} | Loss: {formatted_loss} | Total Improvement: {formatted_total_improvement} | Epochs: {epoch:>{epochs_formatting_length}}/{epochs}")

                # If there is no alert frequency, assume they want all alerts
                else:
                    # Print the progress sheet
                    print(f"Training: {pbar} {formatted_percentage} | Loss: {formatted_loss} | Total Improvement: {formatted_total_improvement} | Epochs: {epoch:>{epochs_formatting_length}}/{epochs}")

            if self.debug:
                print(f"Outputs: {inputs}") # Outputs names as 'inputs' due to feedforward loop's nature
