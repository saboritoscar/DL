#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def run_train(step):
  #read data from file
  data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
  #FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
  data = data_input[0]
  #print ( N.shape(data[0][0])[0] )
  #print ( N.shape(data[0][1])[0] )
  
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
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  y_ = tf.placeholder(tf.float32, [None, 10])
  
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  
  train_step = tf.train.AdagradOptimizer(step).minimize(cross_entropy)
  
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  #TRAINING PHASE
  print("TRAINING")

  accs=[]
  losses=[]

  for i in range(500):
    batch_xs = data[0][0][100*i:100*i+100]
    batch_ys = real_output[100*i:100*i+100]
    sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: data[2][0], y_: real_check})
    accs.append(acc)    
    losses.append(loss)
  
  
  #CHECKING THE ERROR
  print("ERROR CHECK")
  
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))
  out = 1
  return accs,losses


accs1, losses1 = run_train(0.5)
accs2, losses2 = run_train(0.05)
accs3, losses3 = run_train(0.005)
accs4, losses4 = run_train(0.0005)

opt = 'mnist_accuracy_adagrad'
#plt.plot(loss1, color='blue')
plt.title('Adagrad test')
plt.plot(accs1, color='red', label='0.5')
plt.plot(accs2, color='green', label='0.05')
plt.plot(accs3, color='purple', label='0.005')
plt.plot(accs4, color='blue', label='0.0005')
plt.legend()
name = opt +'.png'
plt.xlabel('iteration number')
plt.ylabel('accuracy function value')
plt.savefig(name)

#opt = 'mnist_loss_adam'
#plt.plot(loss1, color='blue')
#plt.title('Adagrad test')
#plt.plot(losses1, color='red', label='0.5')
#plt.plot(losses2, color='green', label='0.05')
#plt.plot(losses3, color='purple', label='0.005')
#plt.plot(losses4, color='blue', label='0.0005')
#plt.legend()
#name = opt +'.png'
#plt.xlabel('iteration number')
#plt.ylabel('accuracy function value')
#plt.savefig(name)

