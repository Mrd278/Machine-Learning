# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:08:28 2020

@author: Mridul Gupta
"""

# Importing the Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation # import animation support
import math

# Generating Hosue Sizes
num_houses = 160
np.random.seed(42)
house_size = np.random.randint(low = 1000, high = 3500, size = num_houses)

# Generating house prices
np.random.seed(42)
house_price = house_size * 100 + np.random.randint(low = 20000, high = 70000,
                                                   size = num_houses)

# Visualizing the data
plt.plot(house_size, house_price, "bx")
plt.xlabel("house_size")
plt.ylabel("house_price")
plt.show()

# Normalizing the data so that  they have the same scale
def normalize(array):
    return (array - array.mean()) / array.std()

# Training data and test data
num_train_samples = math.floor(num_houses * 0.7)
num_test_samples = num_houses - num_train_samples

# Training Data
train_house_size = np.asarray(house_size[: num_train_samples])
train_house_price = np.asarray(house_price[: num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

# Testing Data
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# Setting up of tensorflow placeholders
tf_house_size = tf.placeholder("float", name = "hosue_size")
tf_house_price = tf.placeholder("float", name = "hosue_price")

# Defining Size_factor and price_offset
tf_size_factor = tf.Variable(np.random.randn(), name = "size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name = "price_offset")

# price prediction formula
tf_price_pred = tf.add(tf.multiply(tf_house_size, tf_size_factor),
                       tf_price_offset)

# Defining the loss function
tf_cost = tf.reduce_sum(tf.pow(tf_house_price - tf_price_pred, 2))/(2*num_train_samples)

learning_rate = 0.1

# Defining the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    display_every = 2
    num_training_iter = 50
    
    # Calculate the number of lines to animate
    fit_num_plots = math.floor(num_training_iter / display_every)
    # add storage of factor and offset values from each epoch
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offsets = np.zeros(fit_num_plots)
    fit_plot_idx = 0
    
    for iteration in range(num_training_iter):
        
        for (x, y) in zip(train_house_size_norm, train_house_price_norm):
            sess.run(optimizer, feed_dict = {tf_house_size: x, tf_house_price: y})
            
        if(iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict = {tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost = ", "{:.9f}".format(c), 
                  "size factor = ", sess.run(tf_size_factor), "price offset = ", sess.run(tf_price_offset))
            
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx = fit_plot_idx + 1
            
    print("Optimization Finished! ")
    training_cost = sess.run(tf_cost, feed_dict = {tf_house_size: 
        train_house_size_norm, tf_house_price: train_house_price_norm})
    print("training cost: ", training_cost, "size_factor: ", 
          sess.run(tf_size_factor), "price offset: ", sess.run(tf_price_offset))
    
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    
    test_house_price_mean = test_house_price.mean()
    test_house_price_std = test_house_price.std()
    
    test_house_size_mean = test_house_size.mean()
    test_house_size_std = test_house_size.std()
    
    train_house_price_mean = train_house_price.mean()
    train_house_price_std = train_house_price.std()
    # Plot the Graph
    '''plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.xlabel("Size")
    plt.ylabel("Price")
    plt.plot(train_house_size, train_house_price, 'go', label = 'Training Data')
    plt.plot(test_house_size, test_house_price, 'mo', label = 'Testing Data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
            (sess.run(tf_size_factor) * train_house_size_norm + 
             sess.run(tf_price_offset)) * train_house_price_std + 
             train_house_price_mean,
            label = "Learned Regression")
    plt.legend(loc = "upper left")
    plt.show()
    
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    
    train_house_price_mean = train_house_price.mean()
    train_house_price_std = train_house_price.std()'''
    
    fig, ax = plt.subplots()
    line, = ax.plot(house_size, house_price)
    
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.title("Gradient Descent Fitting Regression Line")
    plt.ylabel("Price")
    plt.xlabel("Size(sq. ft.)")
    plt.plot(train_house_size, train_house_price, 'go', label = 'training data')
    plt.plot(test_house_size, test_house_price, 'mo', label = 'testing data')
    plt.legend(loc = "upper left")
    
    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean)
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offsets[i])
        * train_house_price_std + train_house_price_mean)
        return line,
    
    def initAnim():
        line.set_ydata(np.zeros(shape = house_price.shape[0]))
        return line,
    
    ani = animation.FuncAnimation(fig, animate, frames = np.arange(0, fit_plot_idx),
                                  init_func = initAnim, interval = 1000,
                                  blit = True)
    
    plt.show()