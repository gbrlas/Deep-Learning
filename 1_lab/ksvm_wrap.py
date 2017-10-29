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

from sklearn.svm import SVC

class SVMWrapper:

    def __init__(self, X, Y_, c=1, g='auto'):
        self.model = SVC(gamma=g, C=c, probability=True)
        self.model.fit(X, Y_)

    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.predict_proba(X)

    def support(self):
        return self.model.support_

np.random.seed(100)

C = 2
n = 10
X, Y_, Yoh_ = sample_gmm_2d(6, 2, 10, one_hot=True)

model = SVMWrapper(X, Y_, c=1, g='auto')
decfun = lambda x: model.get_scores(x)[:,1]
probs = model.get_scores(X)
Y = probs.argmax(axis=1)

print("SVM")
bbox=(np.min(X, axis=0), np.max(X, axis=0))
graph_surface(decfun, bbox, offset=0.5)
graph_data(X, Y_, Y, model.support())
plt.show()

print("Accuracy:", accuracy_score(Y, Y_))
cnf_matrix = confusion_matrix(Y, Y_)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=range(0, 2))


config = [X.shape[1], 10, 10, 10, Yoh_.shape[1]]


nn = TFDeep(config, 0.1, 1e-4, tf.nn.sigmoid )
nn.train(X, Yoh_, 10000, printable=False)

probs = nn.eval(X)
Y = probs.argmax(axis=1)

print("DNN")
decfun = lambda x: nn.eval(x)[:,1]
bbox=(np.min(X, axis=0), np.max(X, axis=0))
graph_surface(decfun, bbox, offset=0.5)
graph_data(X, Y_, Y)
plt.show()

get_accuracy(nn, 2, X, Y_)