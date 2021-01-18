# MNIST-classifier
A MLP to classify handwritten digits using the MNIST database. Included program to draw and feed the network new digits.
- Achieves 99% accuracy on the test set after 30 epochs with two hidden layers of 500 and 1000 neurons each
- Applies elastic and affine distortions to the training set at the beginning of each epoch
# Setup
- setup.py installs relevant libraries
- download-samples.py downloads the training and testing data
# Use
- Running main.py loads non-CUDA accelerated network and runs the 'Number Guesser'
- Drawn image will feed through the network when left click is lifted
- Pressing any key clears the screen.
# Requirements
- Python 3.5 or later
# References
- http://neuralnetworksanddeeplearning.com/chap2.html
- https://arxiv.org/pdf/1003.0358.pdf
- http://yann.lecun.com/exdb/mnist/
