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
    correct_scores = tf.gather_nd(model.scores,
                                  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
    ###############################################################################
    # TODO: Produce the saliency maps over a batch of images.                     #
    #                                                                             #
    # 1) Compute the “loss” using the correct scores tensor provided for you.     #
    #    (We'll combine losses across a batch by summing)                         #
    # 2) Use tf.gradients to compute the gradient of the loss with respect        #
    #    to the image (accessible via model.image).                               #
    # 3) Compute the actual value of the gradient by a call to sess.run().        #
    #    You will need to feed in values for the placeholders model.image and     #
    #    model.labels.                                                            #
    # 4) Finally, process the returned gradient to compute the saliency map.      #
    ###############################################################################
    # compute loss
    loss = correct_scores
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=model.labels, 
    #                                                              logits=correct_scores))
    
    # compute the gradient of the loss w.r.t. the image
    dimage = tf.gradients(loss, model.image)
    
    # compute actual value of the gradient
    dimage_values = sess.run(dimage, feed_dict={model.image:X, model.labels:y})[0]              
    
    # compute the saliency map
    saliency = np.abs(dimage_values).max(axis=3)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency




def create_class_visualization(target_y, model, **kwargs):
	"""
	Generate an image to maximize the score of target_y under a pretrained model.

	Inputs:
	- target_y: Integer in the range [0, 1000) giving the index of the class
	- model: A pretrained CNN that will be used to generate the image

	Keyword arguments:
	- l2_reg: Strength of L2 regularization on the image
	- learning_rate: How big of a step to take
	- num_iterations: How many iterations to use
	- blur_every: How often to blur the image as an implicit regularizer
	max_jitter: How much to gjitter the image as an implicit regularizer
	- show_every: How often to show the intermediate result
	"""
	l2_reg = kwargs.pop('l2_reg', 1e-3)
	learning_rate = kwargs.pop('learning_rate', 25)
	num_iterations = kwargs.pop('num_iterations', 300)
	blur_every = kwargs.pop('blur_every', 10)
	max_jitter = kwargs.pop('max_jitter', 16)
	show_every = kwargs.pop('show_every', 25)

	# We use a single image of random noise as a starting point
	X = 255 * np.random.rand(224, 224, 3)
	X = preprocess_image(X)[None]

	########################################################################
	# TODO: Compute the loss and the gradient of the loss with respect to  #
	# the input image, model.image. We compute these outside the loop so   #
	# that we don't have to recompute the gradient graph at each iteration #
	#                                                                      #

	# Note: loss and grad should be TensorFlow Tensors, not numpy arrays!  #
	#                                                                      #
	# The loss is the score for the target label, target_y. You should     #
	# use model.scores to get the scores, and tf.gradients to compute  #
	# gradients. Don't forget the (subtracted) L2 regularization term!     #
	########################################################################
	loss = None # scalar loss
	grad = None # gradient of loss with respect to model.image, same size as model.image
	loss = model.scores[0][target_y] - l2_reg * tf.nn.l2_loss(model.image)
	grad = tf.gradients(loss, model.image)
	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################


	for t in range(num_iterations):
		# Randomly jitter the image a bit; this gives slightly nicer results
		ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
		X = np.roll(np.roll(X, ox, 1), oy, 2)

		########################################################################
		# TODO: Use sess to compute the value of the gradient of the score for #
		# class target_y with respect to the pixels of the image, and make a   #
		# gradient step on the image using the learning rate. You should use   #
		# the grad variable you defined above.                                 #
		#                                                                      #
		# Be very careful about the signs of elements in your code.            #
		########################################################################
		dx = sess.run(grad, feed_dict={model.image:X})[0]
		X = X + learning_rate * dx / np.linalg.norm(dx)
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		# Undo the jitter
		X = np.roll(np.roll(X, -ox, 1), -oy, 2)

		# As a regularizer, clip and periodically blur
		X = np.clip(X, -SQUEEZENET_MEAN/SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN)/SQUEEZENET_STD)
		if t % blur_every == 0:
			X = blur_image(X, sigma=0.5)

		# Periodically show the image
		if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
			plt.imshow(deprocess_image(X[0]))
			class_name = class_names[target_y]
			plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
			plt.gcf().set_size_inches(4, 4)
			plt.axis('off')
			plt.show()
	return X