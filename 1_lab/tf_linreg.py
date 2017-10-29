import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from numpy.random import normal
from scipy import stats as stats
import tensorflow as tf

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pdb

f = lambda x: -2*x + 5

# generiranje više primjera
np.random.seed(42)
Xs = np.random.uniform(-10, 10, 50)
Ys = f(Xs) + normal(0, 10, len(Xs))
Xs, Ys = Xs.reshape(-1, 1), Ys.reshape(-1, 1)
N = len(Xs)

# podatci i parametri
X  = tf.placeholder(tf.float32, [None, 1])
Y_ = tf.placeholder(tf.float32, [None, 1])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a*X + b

# kvadratni gubitak
loss  = (1./(2*N)) * (Y-Y_)**2

trainer = tf.train.GradientDescentOptimizer(0.01)
train_op = trainer.minimize(loss)
grads_and_vars = trainer.compute_gradients(loss, [a, b])
optimize = trainer.apply_gradients(grads_and_vars)
grads_and_vars = tf.Print(grads_and_vars, [grads_and_vars], 'Status:')

grad_a = (1/N) * tf.matmul(Y - Y_, X, transpose_a=True)
grad_b = (1/N) * tf.reduce_sum(Y - Y_)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        val_loss, val_grads, da, db = sess.run([loss, grads_and_vars, grad_a, grad_b], feed_dict={X: Xs, Y_: Ys})
        sess.run(train_op, feed_dict={X: Xs, Y_: Ys})
        val_a, val_b= sess.run([a, b], feed_dict={X: Xs, Y_: Ys})

        if i% 100 == 0:
            print("a:", val_a, "b:", val_b, "loss:", val_loss.sum())
            print(val_grads)
            print(da[0][0], db)
            print()
            
    plt.scatter(Xs, Ys, marker='o')
    plt.plot(Xs, val_a*Xs + val_b, '-')
    plt.show()

