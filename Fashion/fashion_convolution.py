import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from decimal import *

mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images/255
test_images = test_images/255

x_test = test_images
y_test = test_labels
x_train = train_images
y_train = train_labels

# input dimensions
num_train, img_rows, img_cols = x_train.shape
depth = 1
x_train = x_train.reshape(x_train.shape[0],
img_rows, img_cols, depth)
x_test = x_test.reshape(x_test.shape[0],
img_rows, img_cols, depth)
input_shape = (img_rows, img_cols, depth)
# number of convolutional filters to use
nb_filters = 32
# pooling size
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

nb_classes = 10
batch_size = 16
nb_epoch = 20

# Create a simple model with pooling and dropout
model = keras.Sequential()
model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size=kernel_size, activation='relu',input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()


model.fit(x_train, y_train, batch_size=batch_size,
epochs=nb_epoch, verbose=1,
validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
