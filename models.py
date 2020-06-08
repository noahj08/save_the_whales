import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
from keras.applications import VGG16,ResNet50,DenseNet121,InceptionV3
import sklearn
import numpy as np
import matplotlib.pyplot as plt


class Model():
    def __init__(self):
        self.model = Sequential()
        self.datagen = ImageDataGenerator()

    def fit(self,X_train, y_train, X_val, y_val, batch_size, epochs):
        self.datagen.fit(X_train)
        history = self.model.fit_generator(self.datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=X_train.shape[0] // batch_size, epochs=epochs, validation_data=(X_val,y_val),verbose=1)
        plt.clf()
        plt.plot(history.history['loss']) 
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(['train','validation'])
        plt.savefig('loss.png')
        

    def save(self, filepath):
        self.model.save(filepath)
        print(f'Model saved at {filepath}')

    def get_confusion(self, X, y):
        y_pred = self.model.predict(X)
        tmp = np.zeros(y_pred.shape)
        tmp[np.arange(y_pred.shape[0]),np.argmax(y_pred,axis=1)] += 1
        con_mat = sklearn.metrics.confusion_matrix(np.argmax(y,axis=1), np.argmax(tmp,axis=1))
        return con_mat

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f'Model loaded from {filepath}')

    def evaluate(self, X, y, batch_size):
        loss, acc, top5_acc = self.model.evaluate(X, y, batch_size=batch_size)
        return loss, acc, top5_acc

    def predict(self, X):
        return self.model.predict(X)

class Pretrained(Model):
    
    def __init__(self, pretrained_model, input_shape, num_classes,lambd=0.05):
        Model.__init__(self)
        regularizer = tf.keras.regularizers.l2(lambd)
        #picking vgg16 as pretrained (base) model https://keras.io/applications/#vgg16
        if pretrained_model == "vgg16":
            conv_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        elif pretrained_model == "resnet50":
            conv_base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        elif pretrained_model == "densenet121":
            conv_base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
        elif pretrained_model == "inceptionv3":
            conv_base = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
        for layer in conv_base.layers:
            layer.trainable = False

        #maybe unfreeze last layer
        conv_base.layers[-2].trainable = False
        conv_base.layers[-1].trainable = True
        #conv_base.layers.pop()
        #conv_base.layers.pop()

        self.model.add(conv_base)
        self.model.add(Flatten())
        self.model.add(Dropout(0.1))
        #self.model.add(BatchNormalization())
        self.model.add(Dense(48, activation='relu')) #64
        self.model.add(Dropout(0.1))
        #self.model.add(BatchNormalization())
        self.model.add(Dense(48, activation='relu')) #48
        self.model.add(Dropout(0.1))
        #self.model.add(BatchNormalization())
        self.model.add(Dense(num_classes, activation='softmax'))
        for layer in self.model.layers:
            for attr in ['activity_regularizer']:
                if hasattr(layer,attr):
                    setattr(layer,attr,regularizer)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                                  optimizer=keras.optimizers.Adam(lr=0.0001),
                                                metrics=['accuracy','top_k_categorical_accuracy'])
        self.model.summary()


class Simple(Model):

    def __init__(self, input_shape, num_classes,model_id=0,lambd=0.05):
        Model.__init__(self)
        regularizer = tf.keras.regularizers.l2(lambd)
        #picking vgg16 as pretrained (base) model https://keras.io/applications/#vgg16
        if model_id == 0:
                        self.model.add(Conv2D(48, kernel_size=(3, 3),
                                         activation='relu',
                                         input_shape=input_shape))
                        self.model.add(Conv2D(48, (3, 3), activation='sigmoid'))
                        self.model.add(MaxPooling2D(pool_size=(2, 2)))
                        self.model.add(Conv2D(48, (5, 5), activation='sigmoid'))
                        self.model.add(MaxPooling2D(pool_size=(3, 3)))
                        #self.model.add(Dropout(0.33))
                        self.model.add(Flatten())
                        self.model.add(Dense(36, activation='sigmoid'))
                        #self.model.add(Dropout(0.33))
                        self.model.add(Dense(36, activation='sigmoid'))
                        self.model.add(Dense(num_classes, activation='softmax'))
            #self.model.add(BatchNormalization())

        for layer in self.model.layers:
            for attr in ['activity_regularizer']:
                if hasattr(layer,attr):
                    setattr(layer,attr,regularizer)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                                  optimizer=keras.optimizers.Adam(lr=0.00001),
                                                metrics=['accuracy','top_k_categorical_accuracy'])
        self.model.summary()


