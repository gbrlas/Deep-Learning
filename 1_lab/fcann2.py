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

def relu_activation(data_array):
    return np.maximum(data_array, 0)

def softmax(output_array):
    output_array -= np.max(output_array)
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def regularization_L2_softmax_loss(reg_lambda, weight1, weight2):
    weight1_loss = 0.5 * reg_lambda * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * reg_lambda * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss

def fcann2_train(X, Y_):
    '''
       Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

       Povratne vrijednosti
        layer1_weights_array, layer2_weights_array, layer1_biases_array, layer2_biases_array
    '''
    
    # labels one-hot encoding
    b = np.zeros((Y_.size, Y_.max()+1))
    b[np.arange(Y_.size),Y_] = 1
    
    Y_ = b
    
    hidden_nodes = 5
    num_labels = Y_.shape[1]
    num_features = X.shape[1]    
    
    layer1_weights_array = np.random.normal(0, 1, [num_features, hidden_nodes]) 
    layer2_weights_array = np.random.normal(0, 1, [hidden_nodes, num_labels]) 

    layer1_biases_array = np.zeros((1, hidden_nodes))
    layer2_biases_array = np.zeros((1, num_labels))

    param_niter = 100001
    param_delta = 0.1
    param_lambda = 1e-3

    for step in range(param_niter):
        input_layer = np.dot(X, layer1_weights_array)
        hidden_layer = relu_activation(input_layer + layer1_biases_array)
        output_layer = np.dot(hidden_layer, layer2_weights_array) + layer2_biases_array
        output_probs = softmax(output_layer)
        
        loss = cross_entropy_softmax_loss_array(output_probs, Y_)
        loss += regularization_L2_softmax_loss(param_lambda, layer1_weights_array, layer2_weights_array)

        output_error_signal = (output_probs - Y_) / output_probs.shape[0]

        error_signal_hidden = np.dot(output_error_signal, layer2_weights_array.T) 
        error_signal_hidden[hidden_layer <= 0] = 0

        gradient_layer2_weights = np.dot(hidden_layer.T, output_error_signal)
        gradient_layer2_bias = np.sum(output_error_signal, axis = 0, keepdims = True)

        gradient_layer1_weights = np.dot(X.T, error_signal_hidden)
        gradient_layer1_bias = np.sum(error_signal_hidden, axis = 0, keepdims = True)
        
        #regularization
        gradient_layer2_weights += param_lambda * layer2_weights_array
        gradient_layer1_weights += param_lambda * layer1_weights_array

        layer1_weights_array -= param_delta * gradient_layer1_weights
        layer1_biases_array -= param_delta * gradient_layer1_bias
        layer2_weights_array -= param_delta * gradient_layer2_weights
        layer2_biases_array -= param_delta * gradient_layer2_bias
        
        if step % 5000 == 0:
            print('Loss at step {0}: {1}'.format(step, loss))

    return layer1_weights_array, layer1_biases_array, layer2_weights_array, layer2_biases_array

def fcann2_classify(X, layer1_weights_array, layer1_biases_array, layer2_weights_array, layer2_biases_array):
    input_layer = np.dot(X, layer1_weights_array)
    hidden_layer = relu_activation(input_layer + layer1_biases_array)
    scores = np.dot(hidden_layer, layer2_weights_array) + layer2_biases_array
    probs = softmax(scores)

    classes = probs.copy()
    
    return classes[:,1]

def fcann2_decfun(l1_w, l1_b, l2_w, l2_b):
    def classify(X):
        return fcann2_classify(X, l1_w, l1_b, l2_w, l2_b)
    return classify

np.random.seed(100)

# get the training dataset
X,Y_ = sample_gmm(6, 2, 10)

# train the model
l1_w, l1_b, l2_w, l2_b = fcann2_train(X, Y_)

# evaluate the model on the training dataset
probs = fcann2_classify(X, l1_w, l1_b, l2_w, l2_b)

decfun = fcann2_decfun(l1_w, l1_b, l2_w, l2_b)
bbox=(np.min(X, axis=0), np.max(X, axis=0))
graph_surface(decfun, bbox, offset=0.5)

# graph the data points
graph_data(X, Y_, probs, special=[])

# show the plot
plt.show()