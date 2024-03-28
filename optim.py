
class SGD():
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, parameters, gradients): 
        return parameters - self.lr * gradients
