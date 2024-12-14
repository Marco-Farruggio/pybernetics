from typing import Tuple, List, Union
import numpy as np
from . import Layers
from . import Optimizers
from . import __version__

class Loop:
    def __init__(
            self,
            optimizer: Optimizers,
            dataset: Tuple[np.ndarray, np.ndarray],
            loss_function,
            layers: List[Union['Layers.Dense', 'Layers.ActivationFunction']],
            epochs: int = 1000,
            announcements: bool = True,
            show_output: bool = True
        ) -> None:

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.layers = layers
        self.X, self.y = dataset  # Unpack the dataset into X and y (Paired dataset)

        bar_length = 16  # Length of the loading bar
        first_iteration = True

        layer_descriptions = []
        for layer in layers:
            if isinstance(layer, Layers.Dense): layer_descriptions.append("Dense")
            elif isinstance(layer, Layers.ActivationFunction): layer_descriptions.append(layer.function.capitalize())
            else: layer_descriptions.append("Unkown")

        formatted_layers = " -> ".join(layer_descriptions)

        for epoch in range(1, epochs + 1):
            # Perform a forward pass
            inputs = self.X

            for layer in layers:
                layer.forward(inputs)
                inputs = layer.output

            # Compute loss
            loss = loss_function.compute(self.layers[-1].output, self.y)
            if first_iteration is True:
                first_loss = loss
                first_iteration = False

            # Perform a backward pass
            loss_function.backward(self.layers[-1].output, self.y)  # Start from the last layer's output
            for layer in reversed(self.layers):
                layer.backward(loss_function.dinputs)

            # Update weights dynamically using gradient descent
            for layer in self.layers:
                if isinstance(layer, Layers.Dense):
                    self.optimizer.update_params(layer)

            if announcements:
                # Update loading bar and print the loss
                percent = (epoch) / epochs
                num_hashes = int(bar_length * percent)
                loading_bar = "#" * num_hashes + "-" * (bar_length - num_hashes)

                # Variables formatted nicely for printing
                formatted_epochs = f"{epoch}/{epochs}"
                formatted_loading_bar = f"[{loading_bar}]"
                formatted_percentage = f"{(epoch / epochs) * 100:0<6.2f}%"[:7]
                formatted_loss = f"{loss:.29f}"[:29]
                formatted_total_improvement = f"{(first_loss - loss):0<16.16f}"[:16]

                # Print the progress sheet
                print(f"""===================================
        Pybernetics v{__version__}        
        Marco Farruggio
===================================
Training Loop Progress:
-----------------------------------
Layers: {formatted_layers}
Epoch: {formatted_epochs}
Loading: {formatted_loading_bar} {formatted_percentage}
Loss: {formatted_loss}
Total Improvement: {formatted_total_improvement}
===================================""")
                
                if show_output:
                    print(inputs)
