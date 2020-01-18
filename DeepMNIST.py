import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(mean - var)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

with tf.name_scope("Mnist_Input"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

with tf.name_scope("Input_reshape"):
    x_image = tf.reshape(x, shape=[-1, 28, 28, 1], name = "x_image")
    tf.summary.image('input_image', x_image, 5)

def Weigth_Variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev = 0.1, name = name)
    return tf.Variable(initial)

def bias_Variable(shape, name = None):
    initial = tf.constant(0.1, shape=shape, name = name)
    return tf.Variable(initial)

def conv2d(x, W, name = None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)

def max_pool_2x2(x, name = None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding = 'SAME', name = name)

# 1st Convolution Layer
# 32 features for each 5x5 patch of the image
with tf.name_scope("Conv1"):
    with tf.name_scope("Weights"):
        W_conv1 = Weigth_Variable([5,5,1,32], name="Weight")
        variable_summaries(W_conv1)
    with tf.name_scope("biases"):
        b_conv1 = bias_Variable([32], name = "bias")
        variable_summaries(b_conv1)

    conv1_wx_b = conv2d(x_image, W_conv1, name="con2d") + b_conv1
    tf.summary.histogram('conv1_wx_b', conv1_wx_b)
    h_conv1 = tf.nn.relu(conv1_wx_b, name = 'relu')
    tf.summary.histogram('h_conv1', h_conv1)

    h_pool1 = max_pool_2x2(h_conv1, name = 'pool')

# 2nd Convolution layer

with tf.name_scope("Conv2"):
    with tf.name_scope("Weight"):
        W_conv2 = Weigth_Variable([5,5,32,64], name = 'Weight')
        variable_summaries(W_conv2)
    with tf.name_scope("biases"):
        b_conv2 = bias_Variable([64], name = 'bias')
        variable_summaries(b_conv2)

    conv2_wx_b = conv2d(h_pool1, W_conv2, name='conv2d') + b_conv2
    tf.summary.histogram('conv2_wx_b', conv2_wx_b)
    h_conv2 = tf.nn.relu(conv2_wx_b, name = 'relu')
    tf.summary.histogram('h_conv2', h_conv2)
    h_pool2 = max_pool_2x2(h_conv2, name = 'pool')

# Fully Connected Layer
with tf.name_scope("FC"):
    W_fc1 = Weigth_Variable([7*7*64, 1024], name='Weight')
    b_fc1 = bias_Variable([1024], name = 'bias')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name = 'reshape')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = 'relu')

# drop some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='dropout')

# Readout Layer
with tf.name_scope("Readout"):
    W_fc2 = Weigth_Variable([1024, 10], name = 'Weight')
    b_fc2 = bias_Variable([10], name='bias')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # Predicted Probability

with tf.name_scope("Cross_Entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)) # Loss Measurement

with tf.name_scope("Loss_optimizer"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # Loss Optimization

with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y_conv,1))
    Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_accuracy", Accuracy)

summarize_all = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

tbwriter = tf.summary.FileWriter("./DeepMNIST_tensorboard", sess.graph)

import time

num_steps = 2000
display_every = 100

Start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    _, summary = sess.run([train_step, summarize_all], feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

    if i%display_every == 0:
        train_accuracy = Accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob : 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time - Start_time, train_accuracy * 100))
        tbwriter.add_summary(summary, i)

end_time = time.time()
print("Total training Time for {0} batches: {1:.2f} seconds".format(i+1, end_time - Start_time))

print("Test Accuracy: {0:.3f}%".format(Accuracy.eval(feed_dict = {x:mnist.test.images, y_: mnist.test.labels, keep_prob : 1.0}) * 100))

sess.close()
