from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from sklearn.preprocessing import normalize

import cPickle

import os
from scipy.io import loadmat

import random as rd
import timeit

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x.T, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1) 
    return L0, L1, output.T
    
def cost(y, y_):
    #y = predicted, y_ = target
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Compute the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''   

    dCdL1 =  y - y_ #(deriv = pred - target)
    
    h_output = np.dot(W1, dCdL1.T)
    h_input = np.multiply(h_output , (1 - L0**2))
    
    #Gradients for w's and b's.
    dCdW1 =  dot(L0, dCdL1)
    dCdb1 = np.sum(dCdL1, axis=0).reshape(-1, 1)
    dCdW0 = np.dot(h_input, x).T
    dCdb0 = np.sum(h_input, axis=1).reshape(-1, 1)  
    
    return dCdW0, dCdW1, dCdb0, dCdb1

def convertToOneHot(y):
    #create a identity matrix
    one_hot_y = np.identity(_NUM_OF_PARTIES_).tolist() #5 Y values... y will be one-hot-encoding
    
    # [1,0,0,0,0] => liberal
    # [0,1,0,0,0] => covservative, etc
    # [0,0,1,0,0] => 
    # [0,0,0,1,0] => 
    # [0,0,0,0,1] => 
    
    res = []
    
    for i in range(len(y)):
        res.append(one_hot_y[y[i]-1])   #y[i] can be one of [1,2,3,4,5]   
    return np.asarray(res)

'''
Take the data and seperate into training and test sets
Use 70% of data for training, 30% for test
'''
def load_data(data):
    num_examples = data.shape[0] # 121 total examples
    num_train = int(num_examples*0.70) #84 train data, 37 test data
    
    index_train = rd.sample(range(num_examples), num_train)
    index_test = []
    for i in range(num_examples):
        if i not in index_train:
            index_test.append(i)

    input_train = data[range(0, num_examples)][:,:-1] #shape: (84,215)instead of 216
    input_test = data[index_test][:,:-1]  #shape: (37, 215)
    
    target_train =  data[index_train][:,-1].astype(int) #(84,1)
    target_test = data[index_test][:,-1].astype(int)  #(37,1)
    
    #normalze input vectors
    input_train = normalize(input_train)
    input_test = normalize(input_test)
    
    return input_train, input_test, convertToOneHot(target_train), convertToOneHot(target_test)
  
def shuffle_dataset(data_X, data_Y):
    combined = zip(data_X, data_Y)
    random.shuffle(combined)
    X, Y = zip(*combined)
    X, Y = np.asarray(X), np.asarray(Y)    
    return X, Y

def InitWb(num_features, num_hiddens, num_outputs):
    '''Initializes w's and b's
    W1: weights between input and hidden layer
    b1 : bias ""
    W2: weights between hidden layer to output
    b2: bias ""
    '''
    
    W1 = 0.01 * np.random.randn(num_features, num_hiddens)
    W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
    b1 = np.zeros((num_hiddens, 1))
    b2 = np.zeros((num_outputs, 1))
    return W1, W2, b1, b2
    
def NeuralNet(X, Y):
    alpha = 3e-1
    momentum = 5e-1
    num_epochs = 2000 #aka # of iterations..
    num_hiddens = 30 #number of hidden layer units
    
    # initialize weights and biases
    W1, W2, b1, b2 = InitWb(X.shape[1], num_hiddens, Y.shape[1])
    dW1 = np.zeros(W1.shape)
    dW2 = np.zeros(W2.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)

    num_features = X.shape[1]

    for epoch in xrange(num_epochs):
        L0, L1, output = forward(X, W1, b1, W2, b2)

        dCdW0, dCdW1, dCdb0, dCdb1 = \
            deriv_multilayer(W1, b1, W2, b2, X, L0, L1, output, Y)

        #update weights
        dW1 = momentum * dW1 - (alpha / num_features) * dCdW0
        dW2 = momentum * dW2 - (alpha / num_features) * dCdW1
        db1 = momentum * db1 - (alpha / num_features) * dCdb0
        db2 = momentum * db2 - (alpha / num_features) * dCdb1        
        
        W1 = W1 + dW1
        W2 = W2 + dW2
        b1 = b1 + db1
        b2 = b2 + db2   
        
        # this works as a progress bar to let you know which epoch/iteration you are on
        # if epoch % 50 == 0:
        #     print "  ", epoch, "th epoch"
         
    return W1, W2, b1, b2   

def getPrediction(X_test, W1, W2, b1, b2):
    correct_count = 0
    
    y = np.identity(_NUM_OF_PARTIES_).tolist()
    L0, L1, output = forward(X_test, W1, b1, W2, b2)
    
    return output

def getScore(predicted_Y, Y_test):
    
    correct_count = 0
    
    pre_y = np.identity(_NUM_OF_PARTIES_).tolist()

    for i in xrange(predicted_Y.shape[0]):
        vote = np.argmax(predicted_Y[i]) #argmax: which has the largest probability
        if all(pre_y[vote] == Y_test[i]):
            correct_count += 1
     
    CC = correct_count / float(Y_test.shape[0]) * 100 #CC is the percentage of correctly classified
    
    return CC
    
if __name__=="__main__":  
    _NUM_OF_PARTIES_ = 5 # 1 to 5 --> Y values --> CON, LIB, NDP, GRN, Other
    
    data = np.loadtxt('data_2015.csv', delimiter=',', skiprows=1) 
    #data.shape => (121, 216) last column is Y
                    # (examples, features)

    X_train, X_test, Y_train, Y_test = load_data(data)
    
    # this shuffles the orders of the examples
    X_tr, Y_tr = shuffle_dataset(X_train, Y_train)
    
    start = timeit.default_timer()
    
    print "Training the model................."
    W1, W2, b1, b2 = NeuralNet(X_tr, Y_tr)
    # output is W * X + b --> looking for weight/bias to find optimal output
    # trusting that this output is optimized
    # W - weight, b - bias

    print "---------------------------------------"
    print "Predict using test data"
    X_te, Y_te = shuffle_dataset(X_test, Y_test)
    Y_pre = getPrediction(X_te, W1, W2, b1, b2)
    print "---------------------------------------"
    
    score = getScore(Y_pre, Y_te)
    print "Model prediction %:", score
    print "---------------------------------------"

    stop = timeit.default_timer()
    print "runtime: ", (stop - start)/60, " minutes"    
    