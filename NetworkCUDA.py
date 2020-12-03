import numpy as np
import cupy as cp
import random, time, math
from mnist import MNIST
from distort import Deformer
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import keras
import json



##### MNIST Classifier (CUDA - MUST HAVE CUPY INSTALLED) #####
'''
Multilayer Perceptron trained/evaluated with the MNIST database of handwritten digits.
60000 training images are expanded using keras' ImageDataGenerator and the elastic_transform 
preprocessing function found at: https://www.kaggle.com/babbler/mnist-data-augmentation-with-elastic-distortion 
and applies affine transformations. Tests after each epoch on the 10000 test images.

With 2 hidden layers, 1000 and 500 neurons, learning rate decaying from 0.5 -> 0.005, lambda = 0.5,
batch size = 10, sigmoid activation function, and 30 epochs the model achieves 99.00-99.15% accuracy on the test set. 

Included methods to save and load networks, and NumberGuesser.py allows for new inputs to be drawn by the user and 
very accurately classifies them
'''

class Layer:

	def __init__(self, n_inputs, n_neurons):

		#Initialize weights in a uniform random distribution between [-0.05, 0.05]
		self.weights = cp.random.uniform(-0.05, 0.05, (n_neurons, n_inputs))
		#Initialize biases at 0
		self.biases = cp.zeros(n_neurons)
		self.change_in_weight = cp.zeros((n_neurons, n_inputs))
		self.change_in_biases = cp.zeros(n_neurons)

	#Feed forward one image through the layer
	def forward(self, features):

		features = cp.array(features)

		self.z = cp.dot(self.weights, features) + self.biases
		self.activation = sigmoid(self.z)



class Network:

	def __init__(self, sizes, training_images, training_labels, test_images, test_labels):

		self.sizes = sizes
		self.layers = []
		self.training_images = training_images
		self.training_labels = training_labels
		self.test_images = test_images
		self.test_labels = test_labels

		for x in range(len(self.sizes) - 1):

			n_inputs = self.sizes[x]
			n_neurons = self.sizes[x+1]

			self.layers.append(Layer(n_inputs, n_neurons))

	#Feed forward one image through the network
	def feedforward(self, X):

		for layer in self.layers:

			layer.forward(X)

			X = layer.activation

		return X

	def evaluate(self, images=None, labels=None):

		if images is None or labels is None:

			images = self.test_images
			labels = self.test_labels

		#Evaluate Model
		correct = 0
		
		for x in range(len(images)):
			result = self.feedforward(images[x])

			if cp.argmax(result) == labels[x]:
				correct += 1

		return correct


	#Backpropagate the error for one input - X features, Y label
	def backProp(self, X, Y):

		zs = []
		activations = []
		activations.append(X)

		for x in range(len(self.layers)):
			self.layers[x].forward(X)
			X = self.layers[x].activation
			zs.append(self.layers[x].z)
			activations.append(X)


		error = output_error(activations[len(activations) - 1], Y, zs[len(zs) - 1])

		for l in range(len(self.layers)):

			idx = ((len(self.layers) - 1) - l)

			self.layers[idx].change_in_weight += change_in_weights(error, activations[idx])
			self.layers[idx].change_in_biases += error

			if(l < len(self.layers) - 1):
				error = backprop_error(self.layers[idx].weights, error, zs[idx - 1])

	#Update weights and biases by calculated cost gradient 
	def update_mini_batch(self, mini_batch, lr, n, lmbda):


		for i in range(len(self.layers)):

			self.layers[i].change_in_weight = cp.zeros(self.layers[i].weights.shape)
			self.layers[i].change_in_biases = cp.zeros(self.layers[i].biases.shape)

		for x, y in mini_batch:
			self.backProp(x, y)
		
		for j in range(len(self.layers)):

			#L2 regularization
			self.layers[j].weights = ((1-((lr*lmbda)/n)) * self.layers[j].weights) - (lr/len(mini_batch) * self.layers[j].change_in_weight)
			self.layers[j].biases -= lr/len(mini_batch) * self.layers[j].change_in_biases




	def SGD(self, epochs, mini_batch_size, lr, lmbda):

		n = len(self.training_images)
		acc = 0
		times = 0
		currentTime = 0

		
		for j in range(epochs):

			# Distort and scale the training images in [0.5, -0.5]
			training_inputs_deformed = cp.array(Deformer.deform_all(cp.asnumpy(self.training_images)))
			training_inputs_deformed = cp.reshape(training_inputs_deformed, (60000, 784))
			training_inputs_scaled = training_inputs_deformed/127.5 - 1.0

			### Zip together and shuffle training data/labels ###
			training_data = list(zip(training_inputs_scaled, self.training_labels))
			random.shuffle(training_data)

			### Construct mini batches of length mini_batch_size ###
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

			batch_count = 0

			for mini_batch in mini_batches:

				start_time = time.time()
				times += 1

				### Backprop through network for one batch ###
				
				self.update_mini_batch(mini_batch, lr, n, lmbda)

				#### Calculate total time and time elapsed ###
				batch_count += 1

				printProgressBar(batch_count, len(mini_batches), prefix=(('Epoch: %s/%s ') % (j + 1, epochs)))

			# Print Accuracy
			test_acc = 'Test Accuracy: %d' % (self.evaluate(self.test_images, self.test_labels))
			print(test_acc)

			### Decay learning rate after each epoch ###
			lr = decay_lr(0.5, 0.005, epochs, j+1)

	def save(self, file_name):

		weights = []
		biases = []
		layers = []

		for i in range(len(self.layers)):

			weights.append(self.layers[i].weights)
			biases.append(self.layers[i].biases)

		layers.append(np.array(weights, dtype=object))
		layers.append(np.array(biases, dtype=object))

		np.save('%s.npy' % format(file_name), layers, allow_pickle=True)

	@staticmethod
	def load(file_name, training_images, training_labels, test_images, test_labels):

		sizes = [784]

		load_layers = np.load('%s.npy' % (file_name), allow_pickle=True)

		for weights in load_layers[0]:

			sizes.append(len(weights))

		net = Network(sizes, training_images, training_labels, test_images, test_labels)

		for x in range(len(net.sizes) - 1):

			net.layers[x].weights = load_layers[0][x]
			net.layers[x].biases = load_layers[1][x]

		return net






			
## HELPER FUNCTIONS ##


def sigmoid(x):

	return 1/(1 + cp.exp(-x))

def sigmoid_prime(x):

	return sigmoid(x) * (1 - sigmoid(x))

### Exponential decay of learning rate from lr_initial to lr_final ###
def decay_lr(lr_initial, lr_final, total_epochs, current_epoch):

	k = math.log(lr_final/lr_initial)

	lr = lr_initial * math.e**((k/total_epochs) * current_epoch)
	return lr


def cost_derivative(activation, expected):

	return (activation - expected)

def output_error(activation, expected, z):

	return cost_derivative(activation, expected)


def change_in_weights(error, activation):

	error = cp.array([error]).reshape(len(error), 1)

	activation = cp.array([activation]).reshape(1, len(activation))

	return cp.dot(error, activation)

def change_in_bias(error):

	return error

def backprop_error(next_weights, next_error, this_zs):

	next_weights = cp.array(next_weights)
	next_error = cp.array(next_error)
	this_zs = cp.array(this_zs)  

	return cp.dot(next_weights.T, next_error) * sigmoid_prime(this_zs)


# Print iterations progress
def printProgressBar (iteration, total, prefix = 'Batches', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

