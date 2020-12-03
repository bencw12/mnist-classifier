from NetworkCUDA import Network
from NumberGuesser import NumberGuesser
from mnist import MNIST
import cupy as cp


mndata = MNIST('samples')
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
test_images = (cp.array(test_images)/127.5) - 1.0
test_labels = cp.array(test_labels)
train_images = (cp.array(images))

train_label_vecs = []

## Change the labels to vectors
for x in labels:

	vec = [0,0,0,0,0,0,0,0,0,0]
	vec[x] = 1
	train_label_vecs.append(vec)

train_label_vecs = cp.array(train_label_vecs)
# INITIALIZE THE NETWORK 
net = Network([784, 1000, 500, 10], train_images, train_label_vecs, test_images, test_labels)

# TRAINING (Achieves 99% Accuracy)
net.SGD(epochs = 30, mini_batch_size = 10, lr = 0.5, lmbda = 0.5)
net.save('network-new')

trained_net = Network.load('network-final', train_images, train_label_vecs,	 test_images, test_labels)

# EVALUATE
#print(net.evaluate())
print(trained_net.evaluate())

# TEST
game = NumberGuesser(trained_net)
game.run()