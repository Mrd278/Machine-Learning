# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:08:20 2019

@author: Mridul
"""

import tensorflow as tf

a = tf.constant(4.5, name = "Constant_a")
b = tf.constant(5.5, name = "Constant_b")
c = tf.constant(7.0, name = "constant_c")
d = tf.constant(11.2, name = "constant_d")

add_n = tf.add(a,c, name = "add_n")
mul = tf.multiply(a,d, name = "mul")
div = tf.div(b,c, name = "div")
fx = tf.add(add_n,tf.multiply(mul,div), name = "fx")

sess = tf.Session()
print(sess.run(fx))

writer = tf.summary.FileWriter("./m2_example1", sess.graph)
writer.close()
sess.close()