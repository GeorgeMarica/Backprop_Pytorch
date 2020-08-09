import network_numpy
import mnist_loader_numpy

training_data, validation_data, test_data = mnist_loader_numpy.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
epochs = 4
learning_rate = 0.5

net = network_numpy.Network([784, 30, 10])
net.sgd(training_data, epochs, learning_rate, test_data=test_data)
