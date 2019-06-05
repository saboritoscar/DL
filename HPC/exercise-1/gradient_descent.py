#!/usr/bin/env python
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def run_train(step, opt):
  # Model parameters
  W = tf.Variable([.3], dtype=tf.float32)
  b = tf.Variable([-.3], dtype=tf.float32)
  # Model input and output
  x = tf.placeholder(tf.float32)
  linear_model = W * x + b
  y = tf.placeholder(tf.float32)
  
  # loss
  loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
  # optimizer
  if(opt =='sgd'):
    optimizer = tf.train.GradientDescentOptimizer(step)
  elif(opt == 'adam'):
    optimizer = tf.train.AdamOptimizer(step)
  elif(opt == 'adagrad'):
    optimizer = tf.train.AdagradOptimizer(step) 
  train = optimizer.minimize(loss)
  
  # training data
  x_train = [1, 2, 3, 4]
  y_train = [0, -1, -2, -3]
  # training loop
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init) # reset values to wrong
  
  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
  
  W_a = []
  b_a = []
  loss_a = []
  for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    W_a.append(curr_W)
    b_a.append(curr_b)
    loss_a.append(curr_loss)

  return W_a,b_a,loss_a

 
#opt = 'sgd'
#opt = 'adam'
opt = 'adagrad'
#W,b,loss1 = run_train(0.1,opt)
#W,b,loss2 = run_train(0.01,opt)
#W,b,loss3 = run_train(0.001,opt)
#W,b,loss4 = run_train(0.0001,opt)
#W,b,loss5 = run_train(0.00001,opt)

opt = 'adam'
W,b,loss2 = run_train(0.1,opt)
opt = 'adagrad'
W,b,loss3 = run_train(1,opt)
opt = 'sgd'
W,b,loss4 = run_train(0.01,opt)
#W,b,loss5 = run_train(0.001,opt)


#plt.plot(loss1, color='blue')
plt.title('Optimizers comparison')
#plt.plot(loss2, color='red', label='0.01')
#plt.plot(loss3, color='green', label='0.001')
#plt.plot(loss4, color='purple', label='0.0001')
#plt.plot(loss5, color='blue', label='0.00001')

plt.plot(loss2, color='red', label='adam 0.1')
plt.plot(loss3, color='green', label='adagrad 1')
plt.plot(loss4, color='purple', label='sgd 0.01')
#plt.plot(loss5, color='blue', label='0.001')

plt.legend()
name = 'final' +'.png'
plt.xlabel('iteration number')
plt.ylabel('loss function value')
plt.savefig(name)





