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

from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_string('data_dir', 
  '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(
  tf.app.flags.FLAGS.data_dir, one_hot=True)

N = mnist.train.images.shape[0]
D = mnist.train.images.shape[1]
C = mnist.train.labels.shape[1]

print("Train:  N:", N, "D:", D, "C:", C)

N = mnist.test.images.shape[0]
D = mnist.test.images.shape[1]
C = mnist.test.labels.shape[1]

print("Test:  N:", N, "D:", D, "C:", C)
print(mnist.train.images[0].shape)
plt.imshow(mnist.train.images[1].reshape(28,28), cmap='gray')
plt.show()

tf.reset_default_graph()
X, Yoh_ = mnist.train.images, mnist.train.labels
config = [X.shape[1], Yoh_.shape[1]]

nn1 = TFDeep(config, 0.1, 1e-8)
nn1.train(X, Yoh_, 4001)

X, Yoh_ = mnist.test.images, mnist.test.labels
get_accuracy(nn1, 10, X, Yoh_.argmax(1))

tf.reset_default_graph()
X, Yoh_ = mnist.train.images, mnist.train.labels
config2 = [X.shape[1], 100, Yoh_.shape[1]]


nn2 = TFDeep(config2, 0.15, 1e-9)
nn2.train(X, Yoh_, 4001)

X, Yoh_ = mnist.test.images, mnist.test.labels
get_accuracy(nn2, 10, X, Yoh_.argmax(1))

tf.reset_default_graph()
X, Yoh_ = mnist.train.images, mnist.train.labels
config3 = [784, 10]

nn3 = TFDeep(config3, 0.1, 1e-4)
nn3.train_mb(X, Yoh_, n_epochs=1000, batch_size=64, train_ratio=0.8, print_step=10)

X, Yoh_ = mnist.test.images, mnist.test.labels
get_accuracy(nn3, 10, X, Yoh_.argmax(1))

probs = nn3.eval(mnist.test.images)
correct_probs = np.sum(probs * mnist.test.labels, axis=1)
worst_sample = correct_probs.argmin()
plt.imshow(mnist.test.images[worst_sample].reshape(28,28), cmap='gray')
plt.show()

print("Prob", correct_probs[worst_sample])
print("Correct label", mnist.test.labels[worst_sample].argmax())

tf.reset_default_graph()
X, Yoh_ = mnist.train.images, mnist.train.labels
config6 = [784, 100, 10]

nn6 = TFDeep(config6, 0.1, 1e-4)
nn6.train_mb(X, Yoh_, n_epochs=1000, batch_size=64, train_ratio=0.8, print_step=10)

X, Yoh_ = mnist.test.images, mnist.test.labels
get_accuracy(nn6, 10, X, Yoh_.argmax(1))

probs = nn6.eval(mnist.test.images)
correct_probs = np.sum(probs * mnist.test.labels, axis=1)
worst_sample = correct_probs.argmin()
plt.imshow(mnist.test.images[worst_sample].reshape(28,28), cmap='gray')
plt.show()

print("Prob", correct_probs[worst_sample])
print("Correct label", mnist.test.labels[worst_sample].argmax())

tf.reset_default_graph()
X, Yoh_ = mnist.train.images, mnist.train.labels
config5 = [784, 10]

nn5 = TFDeep(config5, 0.1, 1e-4)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.1, global_step, 1, 1 - 1e-4, staircase=True)
nn5.train_step = tf.train.AdamOptimizer(learning_rate).minimize(nn5.loss, global_step=global_step)

nn5.train(X, Yoh_, 1001)

X, Yoh_ = mnist.test.images, mnist.test.labels
get_accuracy(nn5, 10, X, Yoh_.argmax(1))

tf.reset_default_graph()
X, Yoh_ = mnist.train.images, mnist.train.labels
config4 = [784, 10]

nn4 = TFDeep(config4, 0.1, 1e-4)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.1, global_step, 1, 1 - 1e-4, staircase=True)
nn4.train_step = tf.train.AdamOptimizer(learning_rate).minimize(nn4.loss, global_step=global_step)

nn4.train_mb(X, Yoh_, n_epochs=1000, batch_size=64, train_ratio=0.8, print_step=10)

X, Yoh_ = mnist.test.images, mnist.test.labels
get_accuracy(nn4, 10, X, Yoh_.argmax(1))

tf.reset_default_graph()
X, Yoh_ = mnist.train.images, mnist.train.labels
config7 = [784, 100, 10]

nn7 = TFDeep(config7, 0.1, 1e-4)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.1, global_step, 1, 1 - 1e-4, staircase=True)
nn7.train_step = tf.train.AdamOptimizer(learning_rate).minimize(nn7.loss, global_step=global_step)

nn7.train_mb(X, Yoh_, n_epochs=1000, batch_size=64, train_ratio=0.8, print_step=10)

X, Yoh_ = mnist.test.images, mnist.test.labels
get_accuracy(nn7, 10, X, Yoh_.argmax(1))

np.random.seed(100)
X, Yoh_ = mnist.train.images, mnist.train.labels
Y_ = Yoh_.argmax(axis=1)

model = SVMWrapper(X, Y_, c=1, g='auto')

X, Yoh_ = mnist.test.images, mnist.test.labels
Y_ = Yoh_.argmax(axis=1)

probs = model.predict(X)
Y = probs

print("Accuracy:", accuracy_score(Y, Y_))
cnf_matrix = confusion_matrix(Y, Y_)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=range(0, 10))