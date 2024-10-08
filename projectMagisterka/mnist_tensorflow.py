# -*- coding: utf-8 -*-
"""mnist_tensorflow.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZMrjNddJvwA1HKtiSPR4ihUiPuEHxSi-
"""

# Tensorflow mnist
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from random import randint

# load dataset | x -> y | x = (28, 28);float , y = int
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train[2], x_train[2].shape)
# print(x_train.shape, x_train[:100])
# print(x_test.shape, x_test[:100])
# print(y_train.shape, y_train[:100])
# print(y_test.shape, y_test[:100])

# process data, normalize values from (0, 255) to (0, 1) and reshape input values
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print(x_train.shape)
# x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
print(x_train.shape)

# model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# check how it works
print(x_test[0].reshape((1, 28, 28, 1)).shape)
print(model.input_shape)
for i in range(randint(2, 200)):
  test = model.predict(x_test[i].reshape((1, 28, 28, 1)))
  print(f'Expected value {y_test[i]}, returned value of probability distribution of each number {model.predict(x_test[i].reshape((1, 28, 28, 1)))},  and interpreted value {np.argmax(test)}')