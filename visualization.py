from data import get_dataset, get_augmented_dataset
from models import Pretrained
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import keras


from net_visualization_tf import compute_saliency_maps
# from net_visualization_tf import jitter, blur_image

# Load saliency maps computation function
def show_saliency_maps(X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()



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
    - max_jitter: How much to gjitter the image as an implicit regularizer
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
        # Randomly jitter the image a bit; this gives slightly nicer result ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
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

def visualize_classes(target_ys):

	model.type(dtype)
	for target_y in target_ys:
		# target_y = 76 # Tarantula
		out = create_class_visualization(target_y, model)

# plt.figure(figsize=(12, 6))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(X[i])
#     plt.title(y[i])
#     plt.axis('off')
# plt.gcf().tight_layout()

X_train, y_train, X_test, y_test = get_dataset(False)


pretrained_model = 'resnet50'#"vgg16"#"resnet50"
model_filepath = 'resnet_1epoch'
new_filepath = f'{pretrained_model}_{datetime.now()}'
input_shape = X_train[0].shape
num_classes = len(y_train[0])
batch_size = 64
epochs = 1

model = Pretrained(pretrained_model, input_shape, num_classes)

if model_filepath == None:
    model.fit(X_train, y_train, batch_size, epochs)
    model.save(new_filepath)
else:
    model.load(model_filepath)

loss, acc = model.evaluate(X_test, y_test, batch_size)
y_pred = model.predict(X_test)
mask = np.arange(5)
show_saliency_maps(X_test, y_pred, mask)
#target_y can be integer or 


