import random
import torch


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [torch.randn(y, 1, requires_grad=True) for y in sizes[1:]]
        self.weights = [torch.randn(y, x, requires_grad=True) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(torch.mm(w, a) + b)
        return a

    def sgd(self, training_data, epochs, eta, test_data=None):
        for j in range(epochs):
            random.shuffle(training_data)
            for x, y in training_data:
                loss = ((y - self.feed_forward(x))**2).mean()
                loss.backward()
                with torch.no_grad():
                    for layer in range(1, self.num_layers):
                        self.biases[-layer].sub_(self.biases[-layer].grad * eta)
                        self.weights[-layer].sub_(self.weights[-layer].grad * eta)
                        self.biases[-layer].grad.zero_()
                        self.weights[-layer].grad.zero_()
            print('Epoch ', j, ' ', self.evaluate(test_data) / len(test_data) * 100, '%')

    def evaluate(self, test_data):
        """Number of test inputs for which the neural network outputs the correct result"""
        test_results = [(torch.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


# Miscellaneous functions
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
