#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
from tensorflow.python.client import device_lib
from time import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

    return gpu_num

#read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
#print ( N.shape(data[0][0])[0] )
#print ( N.shape(data[0][1])[0] )
gpu_num = check_available_gpus()
keep_prob = tf.placeholder(tf.float32)
#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  


#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0


#set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y_ = tf.placeholder(tf.float32, [None, 10])

losses = []
accs =[]
X_A = tf.split(x, int(gpu_num))
Y_A = tf.split(y_, int(gpu_num))

#declare weights and biases
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def model(X, reuse=False):
  X = tf.reshape(X, [-1, 28, 28, 1])
  with tf.variable_scope('L1', reuse=reuse):
    L1 = tf.layers.conv2d(X, 64, [3, 3], reuse=reuse)
    L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
    L1 = tf.layers.dropout(L1, keep_prob, True)

  with tf.variable_scope('L2', reuse=reuse):
    L2 = tf.layers.conv2d(L1, 128, [3, 3], reuse=reuse)
    L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
    L2 = tf.layers.dropout(L2, keep_prob, True)

  with tf.variable_scope('L2-1', reuse=reuse):
    L2_1 = tf.layers.conv2d(L2, 128, [3, 3], reuse=reuse)
    L2_1 = tf.layers.max_pooling2d(L2_1, [2, 2], [2, 2])
    L2_1 = tf.layers.dropout(L2_1, keep_prob, True)

  with tf.variable_scope('L3', reuse=reuse):
    L3 = tf.contrib.layers.flatten(L2_1)
    L3 = tf.layers.dense(L3, 1024, activation=tf.nn.relu)
    L3 = tf.layers.dropout(L3, keep_prob, True)

  with tf.variable_scope('L4', reuse=reuse):
    L4 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
 
  with tf.variable_scope('LF', reuse=reuse):
    conv_y = tf.layers.dense(L4, 10, activation=None)

  return conv_y


#Crossentropy
#cross_entropy = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

for gpu_id in range(int(gpu_num)):
  with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
    with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
      cost = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=model(X_A[gpu_id],gpu_id > 0),
                                                    labels=Y_A[gpu_id])
      losses.append(cost)
      correct_prediction = tf.equal(tf.argmax(model(X_A[gpu_id], True), 1), tf.argmax(Y_A[gpu_id], 1))
      accuracy= tf.cast(correct_prediction, tf.float32)
      accs.append(accuracy)

cross_entropy= tf.reduce_mean(tf.concat(losses, axis=0))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy, colocate_gradients_with_ops=True)



accuracy = tf.reduce_mean(tf.concat(accs, axis=0))

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
  sess.run(tf.global_variables_initializer())
  #TRAIN 
  print("TRAINING")
  start_time = time()
  epochs=10
  gpu_num= int(gpu_num)
  for e in range(epochs):
    for i in range(int(62/gpu_num)):
  
      #until 1000 96,35%
      batch_ini = 800*gpu_num*i
      batch_end = 800*gpu_num*i+800*gpu_num
  
  
      batch_xs = data[0][0][batch_ini:batch_end]
      batch_ys = real_output[batch_ini:batch_end]
      #batch_xs = batch_xs.reshape(-1, 28, 28, 1)

      #train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  
      if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print('step %d, training accuracy %g Batch [%d,%d]' % (i, train_accuracy, batch_ini, batch_end))
  
      sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  print("Train time:")
  end_time = time()
  print(end_time - start_time)
  #TEST
  print("TESTING")

  train_accuracy = accuracy.eval(feed_dict={x: data[2][0], y_: real_check, keep_prob: 1.0})
  print('test accuracy %g' %(train_accuracy))




