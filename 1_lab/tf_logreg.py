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

class TFLogreg:
    def __init__(self, D, C, param_delta=0.01, param_lambda = 0.02):
    # definicija podataka i parametara:
    # definirati self.X, self.Yoh_, self.W, self.b
        self.X  = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.W = tf.Variable(tf.random_normal([D, C]))
        self.b = tf.Variable(tf.random_normal([1, C]))
        self.learning_rate = param_delta
        self.regularization_lambda = param_lambda

    # formulacija modela: izračunati self.probs
    #   koristiti: tf.matmul, tf.nn.softmax
        self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)

    # formulacija gubitka: self.loss
    #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean        
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.Yoh_*tf.log(self.probs), reduction_indices=1))
        self.regularizer = self.regularization_lambda * tf.nn.l2_loss(self.W)
        self.loss = self.cross_entropy + self.regularizer

    # formulacija operacije učenja: self.train_step
    #   koristiti: tf.train.GradientDescentOptimizer,
    #              tf.train.GradientDescentOptimizer.minimize
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    # instanciranje izvedbenog konteksta: self.session
    #   koristiti: tf.Session
        self.session = tf.Session()

    def train(self, X, Yoh_, param_niter):
    # incijalizacija parametara
    #   koristiti: tf.initialize_all_variables
        self.session.run(tf.global_variables_initializer())

    # optimizacijska petlja
    #   koristiti: tf.Session.run
        for i in range(param_niter):
            self.session.run(self.train_step, feed_dict={self.X:X, self.Yoh_:Yoh_})
            if i % 1000 == 0:
                loss = self.session.run(self.loss, feed_dict={self.X: X, self.Yoh_:Yoh_})
                print("{0:4}. Loss: {1:.8f}".format(i, loss))

    def eval(self, X):
    #   koristiti: tf.Session.run
        probs = self.session.run(self.probs, feed_dict={self.X: X})
        return probs
    
def evaluate_performance(Y, Y_):   
    accuracy = accuracy_score(Y_, Y)
    precision = precision_score(Y_, Y)
    recall = recall_score(Y_, Y)
    f1 = f1_score(Y_, Y)
    print("Accuracy: {0:.3f}\n"
          "Precision: {1:.3f}\n"
          "Recall: {2:.3f}\n"
          "F1: {3:.3f}".format(accuracy, precision, recall, f1))
        
def tflogreg_decfun(tflogreg):
    def classify(X):
        return tf_classify(X, tflogreg)
    return classify
    
def tf_classify(X, tflogreg):
        return np.argmax(tflogreg.eval(X), axis=1)

np.random.seed(100)
tf.set_random_seed(100)

# instanciraj podatke X i labele Yoh_
X,Y_ = sample_gmm(4, 3, 10)
b = np.zeros((Y_.size, Y_.max()+1))
b[np.arange(Y_.size),Y_] = 1
Yoh_ = b

# izgradi graf:
tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.01)

# nauči parametre:
tflr.train(X, Yoh_, 20000)

# dohvati vjerojatnosti na skupu za učenje
probs = tflr.eval(X)

# ispiši performansu (preciznost i odziv po razredima)
eval_perf_multi(tf_classify(X, tflr), Y_)

# iscrtaj rezultate, decizijsku plohu
decfun = tflogreg_decfun(tflr)
bbox=(np.min(X, axis=0), np.max(X, axis=0))
graph_surface(decfun, bbox, offset=0.5)

# graph the data points
graph_data(X, Y_, tf_classify(X, tflr), special=[])

# show the plot
plt.show()