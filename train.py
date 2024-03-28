
import numpy as np
import matplotlib.pyplot as plt
import model as m
import optim
import mnist
from loss import LogLoss, ZeroOneLoss

def train_loop(dataset, model, optimizer, loss_fn):

    mean_loss = 0
    for i, (x, label) in enumerate(dataset):
        pred = model.forward(x) # Make a prediction
        loss = loss_fn.compute_loss(pred, label)
        mean_loss += loss
        # print(f'Sample {i+1}, loss: {loss}')
        gradients = loss_fn.compute_gradients(x)
        model.parameters = optimizer.step(model.parameters, gradients) # Update the model parameters

    return mean_loss / len(dataset) # Return mean loss per epoch

def test_loop(dataset, model, loss_fn): # Answer for last part of 3(c)

    mean_loss = 0
    for x, label in dataset:
        pred = model.forward(x)
        loss = loss_fn.compute_loss(pred, label)
        mean_loss += loss

    return mean_loss / len(dataset)


def main():
    train_images = mnist.load_images('mnist/train-images-idx3-ubyte')
    train_labels = mnist.load_labels('mnist/train-labels-idx1-ubyte')

    test_images = mnist.load_images('mnist/t10k-images-idx3-ubyte')
    test_labels = mnist.load_labels('mnist/t10k-labels-idx1-ubyte')
    
    train_dataset = list(zip(train_images, train_labels))
    test_dataset = list(zip(test_images, test_labels))

    lr_losses = [] 
    for lr in [1e-8, 1e-7, 1e-6, 1e-5]:

        model = m.Linear(784, 10)
        loss_fn = LogLoss() # For training
        test_loss_fn = ZeroOneLoss() # For testing
        optimizer = optim.SGD(lr=lr)

        print(f'\n======================== Learning rate: {lr} ========================')
        mean_losses = [] # Store mean losses for each epoch
        for epoch in range(10):
            mean_loss = train_loop(train_dataset, model, optimizer, loss_fn)
            mean_losses.append(mean_loss)
            mean_test_loss = test_loop(test_dataset, model, test_loss_fn)
            np.save(f'models/linear_lr_{lr}_epoch_{epoch+1}', model.parameters)
            print(f'Epoch {epoch+1}, mean loss: {mean_loss}, mean test zero-one loss: {mean_test_loss}')
        lr_losses.append(mean_losses) # Store array of mean losses for each learning rate, for plotting
    
    # Loss curves for different learning rates
    plt.figure()
    for i, lr in enumerate([1e-8, 1e-7, 1e-6, 1e-5]):
        plt.plot(lr_losses[i], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    # plt.show()

if __name__ == '__main__':
    main()