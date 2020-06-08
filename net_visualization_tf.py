import tensorflow as tf
import random
import numpy as np
# from .image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

# matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def compute_saliency_maps(X, y, model):
	"""
	Compute a class saliency map using the model for images X and labels y.

	Input:
	- X: Input images, numpy array of shape (N, H, W, 3)
	- y: Labels for X, numpy of shape (N,)
	- model: A SqueezeNet model that will be used to compute the saliency map.

	Returns:
	- saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
	input images.
	"""
	saliency = None
	# Compute the score of the correct class for each example.
	# This gives a Tensor with shape [N], the number of examples.
	#
	# Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
	# for computing vectorized losses.

	###############################################################################
	# TODO: Produce the saliency maps over a batch of images.                     #
	#                                                                             #
	# 1) Define a gradient tape object and watch input Image variable             #
	# 2) Compute the “loss” for the batch of given input images.                  #
	#    - get scores output by the model for the given batch of input images     #
	#    - use tf.gather_nd or tf.gather to get correct scores                    #
	# 3) Use the gradient() method of the gradient tape object to compute the     #
	#    gradient of the loss with respect to the image                           #
	# 4) Finally, process the returned gradient to compute the saliency map.      #
	###############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	y = tf.cast(y, dtype=tf.int32)
	X_temp = tf.cast(tf.Variable(X), dtype=tf.float32)

	with tf.GradientTape() as tape:
		tape.watch(X_temp)
		scores = model(X_temp)
		correct = tf.gather_nd(scores,tf.stack((tf.range(X.shape[0]), y), axis=1))
		# print(correct)
		# loss = tf.math.reduce_sum(correct)
		# print(loss)
	dx = tape.gradient(correct, X_temp)
	saliency = np.max(np.abs(dx), axis=3)



	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	##############################################################################
	#                             END OF YOUR CODE                               #
	##############################################################################
	return saliency



