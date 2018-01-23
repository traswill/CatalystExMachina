import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def basic_setup():
    var1 = tf.constant(5, name='var1')
    var2 = tf.constant(3, name='var2')
    varsum = tf.add(var1, var2, name='add')

    with tf.Session() as sess:
        # writer creates files for tensorboard
        # get to by running tensorboard --logdir='.//TFgraphs'

        # writer = tf.summary.FileWriter('.\\TFgraphs', sess.graph)
        print(sess.run(varsum))

def linear_regression():
    # 1. Create Data
    xdat = np.random.randint(low=5, high=35, size=30)
    ydat = xdat * np.random.randint(0,100,30)/100 + np.random.randint(-5,5,30)
    plt.scatter(xdat, ydat)
    # plt.show()

    # 2. Define Placeholders
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")

    # 3. Define Variables
    w = tf.Variable(0.0, name="weights")
    b = tf.Variable(0.0, name="bias")

    # 4. Construct Model
    Y_predicted = X * w + b

    # 5. Construct Loss Function for optimization
    loss = tf.square(Y - Y_predicted, name="loss")



if __name__ == '__main__':
    # basic_setup()
    linear_regression()