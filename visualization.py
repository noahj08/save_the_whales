from data import get_dataset, get_augmented_dataset, get_imgs_from_idx
from models import Pretrained
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import keras
from keras.models import load_model
from vis.visualization import visualize_saliency

augmented = False

if augmented:
    X_train, y_train, X_test, y_test = get_augmented_dataset(n_aug=5)
else:
    X_train, y_train, X_test, y_test = get_dataset(False)
pretrained_model = 'vgg16'#"vgg16"#"resnet50"
model_filepath =  'saved_models/vgg1'
new_filepath = f'saved_models/{pretrained_model}_{datetime.now()}'
input_shape = X_train[0].shape
num_classes = len(y_train[0])
batch_size = 64
epochs = 2


model = Pretrained(pretrained_model, input_shape, num_classes)

if model_filepath == None:
	model.fit(X_train, y_train, X_test, y_test, batch_size=batch_size, epochs=epochs)
	model.save(new_filepath)
else:
	# model = load_model(model_filepath, custom_objects={'top_k_categorical_accuracy':tf.keras.metrics.TopKCategoricalAccuracy})
	model.model.load_weights(model_filepath)

filter_indices = [0,1]
classes = tf.math.argmax(model.predict(X_test), axis=1)
classes = np.array(tf.gather(classes, filter_indices))

imgs, labels = get_imgs_from_idx(filter_indices)
for i in range(len(filter_indices)):
	plt.subplot(2, len(filter_indices), i + 1)
	plt.title(str(labels[i]))
    
	plt.imshow(imgs[i], cmap='gray')
	img = visualize_saliency(model.model, -1, filter_indices[i], X_test, wrt_tensor=None)
	plt.subplot(2, len(filter_indices), len(filter_indices)+i+1)
	# plt.title('Saliency for Class ' +str(classes[i]))
	plt.imshow(img, cmap=plt.cm.hot)
	plt.axis('off')        

# print(pretrained_model+" class "+str(idx))
plt.gcf().tight_layout()
plt.savefig(pretrained_model+'_saliency')



