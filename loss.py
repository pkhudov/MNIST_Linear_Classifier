import numpy as np

class LogLoss():
    def __init__(self):
        self.pred = None
        self.label = None

    def compute_loss(self, pred, label):
        self.label = label
        self.pred = pred
        lse = np.max(pred) + np.log(np.sum(np.exp(pred - np.max(pred)))) ## LogSumExp trick applied to log(sum(exp(pred))
        return -pred[label] + lse # Log loss
    
    def compute_gradients(self, x):
        x = np.append(x, 1) # Add bias term
        
        shifted = self.pred - np.max(self.pred) # Doesn't matter, because Softmax(x) = Softmax(x + c) for any c. Will prevent overflow
        softmax= np.exp(shifted) / np.sum(np.exp(shifted))
        gradients = np.outer(x, softmax) # Compute as if all labels are incorrect
        gradients[:, self.label] -= x # Correct the gradient for the true class
        return gradients # Shape [input_dim+1, num_classes]

class ZeroOneLoss(): # For testing. Answer for last part of 3(c)
    def compute_loss(self, pred, label):
        return 0 if np.argmax(pred) == label else 1
