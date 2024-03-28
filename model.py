import numpy as np

class Linear():
    def __init__(self, input_dim, num_classes):
        self.parameters = np.random.randn(input_dim+1, num_classes) * 0.01 # Randomly initialise parameters to be small

    def forward(self, x):
        x = np.append(x, 1) # Add bias term
        return np.dot(x, self.parameters) # Shape [num_classes]
    
