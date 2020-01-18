# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:40:17 2019

@author: Mridul
"""
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

(X_train,y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(10000,784)
X_test = X_test.astype('float32')

X_test = X_test/255

y_test = np_utils.to_categorical(y_test, 10)

mnist_model = load_model('keras_mnist.h5')

loss_and_metrics = mnist_model.evaluate(X_test, y_test, verbose = 2)

print('Test Loss: {}'.format(loss_and_metrics[0]),'Accuracy: {}'.format(loss_and_metrics[1]))

plt.imshow(X_test[1,:].reshape(28,28), cmap = 'gray')

predicted_classes = mnist_model.predict_classes(X_test)

Y_test = np.zeros(10000)

for i in range(10000):
    for j in range(10):
        if y_test[i,j] == 1:
            Y_test[i] = j
            
Y_test = Y_test.astype('int64')

correct_indices = np.nonzero(Y_test == predicted_classes)
print('Correct Predictions: {}'.format(np.count_nonzero(correct_indices)))