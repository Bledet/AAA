import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from decimal import *

mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images/255
test_images = test_images/255
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

model = keras.Sequential([
# Le modèle Sequential est un ensemble linéaire de couches
keras.layers.Flatten(input_shape=(28,28)),
# Transforme une matrice 28x28 en un tableau de 784
keras.layers.Dense(128, activation=tf.nn.relu),
# Couche entièrement connectée de 128 neurones
keras.layers.Dense(10, activation=tf.nn.softmax)
# Couche entièrement connectée de  10 neurones:
#    10 probabilités de sortie
])

model.compile(optimizer=
'sgd',
# On choisit la descente de gradient
# stochastique commme optimisation
loss='sparse_categorical_crossentropy',
# Définition de la mesure de perte
# Ici l'entropie croiée
metrics=['accuracy']
# Définition de la mesure de performance
# que l'on souhaite utiliser. Ici la accuracy
)

model.fit(train_images, train_labels, epochs=50)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("perte: {}, accuracy: {}".format(test_loss, test_acc))
