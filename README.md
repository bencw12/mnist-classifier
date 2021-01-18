# MNIST-classifier
A MLP to classify handwritten digits using the MNIST database. Included program to draw and feed the network new digits.
- Achieves 99% accuracy on the test set after 30 epochs with two hidden layers of 500 and 1000 neurons each
- Applies elastic and affine distortions to the training set at the beginning of each epoch
- To use the Number Guesser:
    - Feeds image through network when left mouse is lifted
    - Press any key to clear screen
- By default, main.py loads non-GPU accelerated network, but CUDA acceleration is supported with NetworkCUDA.py and network-CUDA.npy
# Requirements
- Python 3.5 or later
# References
- http://neuralnetworksanddeeplearning.com/chap2.html
- https://arxiv.org/pdf/1003.0358.pdf
- http://yann.lecun.com/exdb/mnist/
