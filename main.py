# Main script for training and testing models
from data import get_dataset, get_augmented_dataset
from models import Pretrained
from datetime import datetime

X_train, y_train, X_test, y_test = get_dataset()

pretrained_model = "inceptionv3"
model_filepath = None
new_filepath = f'{pretrained_model}_{datetime.now()}'
input_shape = X_train[0].shape
num_classes = len(y_train[0])
batch_size = 16
epochs = 10

model = Pretrained(pretrained_model, input_shape, num_classes)

# Setting weights
if model_filepath == None:
    model.fit(X_train, y_train, batch_size, epochs)
    model.save(new_filepath)
else:
    model.load(model_filepath)

# Evaluation
loss, acc = model.evaluate(X_test, y_test, batch_size)
y_pred = model.predict(X_test)
print(f'Loss = {loss}')
print(f'Accuracy = {acc}')
print(f'Predictions = {y_pred}')
