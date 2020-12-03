import keras
from mnist import MNIST
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from matplotlib import pyplot as plt


def elastic_transform(image, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       
   # Arguments
       image: Numpy array with shape (height, width, channels). 
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """
    
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


class Deformer(object):


	@staticmethod
	def deform_all(images, alpha_range = [8, 10], sigma = 3, cuda=False):


		images1 = np.ones((60000, 28, 28))

		for x in range(len(images)):

			images1[x] = np.reshape(images[x], (-1, 28))

		images1 = np.expand_dims(images1[x in range(60000)], -1)

    #lambda x: elastic_transform(x, alpha_range = alpha_range, sigma=sigma)

		datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=2, height_shift_range=2, zoom_range=0.2, preprocessing_function=lambda x: elastic_transform(x, alpha_range = alpha_range, sigma=sigma), rotation_range = 10, shear_range = 10)

		x = [datagen.flow(images1[0], shuffle=False, batch_size = 60000).next() for i in range(1)]
		x = np.squeeze(np.concatenate(x, axis=-1), 3)
    
		return x
















