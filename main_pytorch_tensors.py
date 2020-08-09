import network_pytorch_tensors
import mnist_loader_pytorch

training_data, validation_data, test_data = mnist_loader_pytorch.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
epochs = 4
learning_rate = 0.5

net = network_pytorch_tensors.Network([784, 30, 10])
net.sgd(training_data, epochs, learning_rate, test_data=test_data)
