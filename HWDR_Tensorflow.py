# Importing the Libraries/Tools for Deep Learning
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''Steps included in Recognition of Handwritten Digits: 
1. Data Preparation [Mnist Data]
2. Inference - Prediction formula [sum(x * weight) + bias -> activation]
3. Loss Calculation [Cross Entropy]
4. Optimisation to Minimize the loss [Gradient Descent Optimizer]

Details about the dataset used: 
Mnist Dataset is being used having total 70,000 pictures
50,000 - training data
15,000 - test data
5,000 - validation data
All the pictures are in Grayscale'''

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # Getting the data from the dataset folder MNIST_data

x = tf.placeholder(tf.float32, shape=[None, 784]) # Input Layer containing 28 X 28 pixel images   

y_ = tf.placeholder(tf.float32, shape=[None, 10]) # Output Layer containing the probability of the digits (0-9)

W = tf.Variable(tf.zeros([784, 10])) # Weights 784 X 10
b = tf.Variable(tf.zeros([10])) # Bias value 1 X 10 

y = tf.nn.softmax(tf.matmul(x,W) + b) # Defining the model (Activation Function used here is softmax)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})

print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()