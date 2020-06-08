# Main script for training and testing models
from data import get_dataset, get_augmented_dataset
from models import Pretrained,Simple
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

augmented=False
if augmented:
    X_train, y_train, X_test, y_test = get_augmented_dataset(n_aug=5)
else:
    X_train, y_train, X_test, y_test = get_dataset()

#from keras.applications import VGG16,ResNet50,DenseNet121,InceptionV3
pretrained_model = "inceptionv3"
model_filepath = None
new_filepath = f'{pretrained_model}_{datetime.now()}'
input_shape = X_train[0].shape
num_classes = len(y_train[0])
batch_size = 16
epochs = 20

#model = Pretrained(pretrained_model, input_shape, num_classes)
model = Simple(input_shape, num_classes, 0)

# Setting weights
if model_filepath == None:
    model.fit(X_train, y_train, X_test, y_test, batch_size, epochs)
    confusion = model.get_confusion(X_train,y_train)
    print(confusion)
    sns.heatmap(confusion/np.sum(confusion))
    plt.savefig(f'{pretrained_model}_confusion.png')
    model.save(new_filepath)
else:
    model.load(model_filepath)

# Evaluation
loss, acc = model.evaluate(X_test, y_test, batch_size)
y_pred = model.predict(X_test)
print(f'Loss = {loss}')
print(f'Accuracy = {acc}')
print(f'Predictions = {y_pred}')
