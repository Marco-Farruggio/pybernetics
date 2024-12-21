from typing import List
from . import Layers
from . import __version__
from ._Typing import Optimizer, LossFunction, Layer, Dataset

class Loop:
    def __init__(
            self,
            optimizer: Optimizer,
            dataset: Dataset,
            loss_function,
            layers: List[Layer],
            epochs: int = 1000,
            alert: bool = True,
            alert_freq: int = 100,
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
            
            # Perform a forward pass through the full network
            for layer in layers:
                layer.forward(inputs)
                inputs = layer.output

            # Compute loss
            loss = loss_function.compute(self.layers[-1].output, self.y)
            if self.first_loss is None:
                self.first_loss = loss

            # Perform a backward pass
            loss_function.backward(self.layers[-1].output, self.y)  # Start from the last layer's output
            for layer in reversed(self.layers):
                layer.backward(loss_function.dinputs)

            # Update weights dynamically using gradient descent
            for layer in self.layers:
                if isinstance(layer, Layers.Dense):
                    self.optimizer.update_params(layer)

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
            
            # If alerts are enabled, print the progress sheet
            if self.alert:
                if self.alert_freq is not None:
                    # If the alert frequency is set, only print the progress sheet at those intervals
                    if epoch % self.alert_freq == 0 or epoch == self.epochs:
                        # Print the progress sheet
                        print(f"Training: {pbar} {formatted_percentage} | Loss: {formatted_loss} | Total Improvement: {formatted_total_improvement} | Epochs: {formatted_epochs}")

                # If there is no alert frequency, assume they want all alerts
                else:
                    # Print the progress sheet
                    print(f"Training: {pbar} {formatted_percentage} | Loss: {formatted_loss} | Total Improvement: {formatted_total_improvement} | Epochs: {formatted_epochs}")
                
            if self.debug:
                print(inputs) # Outputs names as 'inputs' due to feedforward loop's nature
