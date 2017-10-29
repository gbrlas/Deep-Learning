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

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def get_accuracy(nn, n_classes, X, Y_):
    probs = nn.eval(X)
    Y = probs.argmax(axis=1)

    print("Accuracy:", accuracy_score(Y, Y_))

    cnf_matrix = confusion_matrix(Y, Y_)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(0, n_classes))

class TFDeep:
    def __init__(self, config, param_delta, param_lambda=1e-4, activation=tf.nn.relu):
        D = config[0]
        C = config[-1]
        n_layers = len(config[1:])

        # data
        self.X  = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        
        activations = [activation] * (n_layers-1) + [tf.nn.softmax]
        self.Ws = []
        self.bs = []

        reg_loss = 0
        prev_out = self.X
        for i, (prev, next) in enumerate(zip(config, config[1:])):
            W = tf.Variable(tf.random_normal([next, prev]), name='W%s' % i)
            b = tf.Variable(tf.random_normal([next]), name='b%s' % i)

            self.Ws.append(W); self.bs.append(b)
            reg_loss += tf.nn.l2_loss(W)
            
            s = tf.add(tf.matmul(prev_out, W, transpose_b=True), b)
                     
            prev_out = activations[i](s)

            

        # output
        self.probs = prev_out
        err_loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_ * tf.log(self.probs + 1e-10), 1))
        self.loss = err_loss + param_lambda * reg_loss

        self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)
        
        self.session = tf.Session()



    def train(self, X, Yoh_, param_niter, printable=True):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        # incijalizacija parametara
        self.session.run(tf.global_variables_initializer())

        # train loop
        for i in range(param_niter):
            val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
            if i % 100 == 0 and printable: 
                print("{}\t{}".format(i, val_loss))
    
    def _shuffle(self, X, Yoh_):
        perm = np.random.permutation(len(X))
        return X[perm], Yoh_[perm]
        
    def _split_dataset(self, X, Yoh_, ratio=0.8):
        X, Yoh_ = self._shuffle(X, Yoh_)
        split = int(ratio * len(X))
        return X[:split], X[split:], Yoh_[:split], Yoh_[split:],
    
    def train_mb(self, X, Yoh_, n_epochs=1000, batch_size=64, train_ratio=0.8, print_step=100):
        self.session.run(tf.global_variables_initializer())
        prev_loss = window_loss = float('inf'); 
        
        X_train, X_val, Y_train, Y_val = self._split_dataset(X, Yoh_, ratio=train_ratio)
        n_samples = len(X_train)
        n_batches = int(n_samples/batch_size)
        
        for epoch in range(n_epochs):
            X_train, Y_train = self._shuffle(X_train, Y_train)
            i = 0
            avg_loss = 0
            
            while i < n_samples:
                batch_X, batch_Yoh_ = X_train[i:i+batch_size], Y_train[i:i+batch_size]
                data_dict = {self.X: batch_X, self.Yoh_: batch_Yoh_}
                val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=data_dict)
                
                avg_loss += val_loss / n_batches
                i += batch_size
            
            # validation
            data_dict = {self.X: X_val, self.Yoh_: Y_val}
            val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=data_dict)
            window_loss = min(window_loss, val_loss)
            
            if epoch % 50 == 0:
                if window_loss > prev_loss:
                    print("Early stopping: epoch", epoch)
                    break
                prev_loss = window_loss
                window_loss = float('inf')
            
            if epoch % print_step == 0:
                print("Epoch: {:4d}; avg_train_loss {:.9f}; validation_loss {:.9f}".format(epoch, avg_loss, val_loss))
    
        print("Optimization Finished!")
        print("Validation loss {:.9f}".format(val_loss))
        
    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        probs = self.session.run(self.probs, {self.X: X})
        return probs

def count_params():
    total = 0
    for var in tf.trainable_variables():
        shape = var.get_shape()
        var_params = int(np.prod(shape))
        print('{}; shape: {}; total: {}'.format(var.name, shape, var_params))
        total += var_params
    print('Total:', total, "\n\n")

np.random.seed(100)
tf.set_random_seed(100)

# instanciraj podatke X i labele Yoh_
X,Y_ = sample_gmm(4, 3, 10)
b = np.zeros((Y_.size, Y_.max()+1))
b[np.arange(Y_.size),Y_] = 1
Yoh_ = b

config9 = [2, 3]

tf.reset_default_graph()
nn9 = TFDeep(config9, 0.1, 1e-4, activation_function)
nn9.train(X, Yoh_, 10000, printable=False)

probs = nn9.eval(X)
Y = probs.argmax(axis=1)

decfun = lambda x: np.argmax(nn9.eval(x), axis=1)
bbox=(np.min(X, axis=0), np.max(X, axis=0))
graph_surface(decfun, bbox, offset=0.5)
graph_data(X, Y_, Y)
plt.show()

count_params()
accuracy_score(Y, Y_)

tf.set_random_seed(42)
np.random.seed(42)


X, Y_, Yoh_ = sample_gmm_2d(4, 2, 10, one_hot=True)

# [2,2]
config1 = [X.shape[1], Yoh_.shape[1]]

# [2,10,2]
config2 = [X.shape[1], 10, Yoh_.shape[1]]

# [2, 10, 10, 2]
config3 = [X.shape[1], 10, 10, 10, Yoh_.shape[1]]

activation_functions = ["ReLU", "Sigmoid"]
configurations = ["[2,2]", "[2,10,2]", "[2, 10, 10, 2]"]

i = 0
for activation_function in [tf.nn.relu, tf.nn.sigmoid]:
    print("Activation function:", activation_functions[i])
    i +=1 
    
    j = 0
    for config in [config1, config2, config3]:
        tf.reset_default_graph()
        
        print("NN configuration:", configurations[j])
        j += 1
        
        nn = TFDeep(config, 0.1, 1e-4, activation_function)
        nn.train(X, Yoh_, 10000, printable=False)

        probs = nn.eval(X)
        Y = probs.argmax(axis=1)
        
        decfun = lambda x: nn.eval(x)[:,1]
        bbox=(np.min(X, axis=0), np.max(X, axis=0))
        graph_surface(decfun, bbox, offset=0.5)
        graph_data(X, Y_, Y)
        plt.show()
        
        count_params()
        get_accuracy(nn, 2, X, Y_)
